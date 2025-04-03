// Copyright 2022 The BladeDISC Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/disc/disc_util.h"
#include "mlir/disc/transforms/PassDetail.h"
#include "mlir/disc/transforms/shape_utils.h"
#include "mlir/disc/utils/cycle_detector.h"

#define DEBUG_TYPE "disc-dot-merge"

// TODO: this pass shares some common functions with lhlo-fusion-pass. The dot
// merge can actually be regarded as a kind of horizontal fusion. We will
// reassess the necessity of merging this pass into the fusion-pass after fixing
// the bugs in horizontal fusion functions in lhlo-fusion-pass.
// 这个pass可以看做是一个横向fusion的pass

namespace mlir {
namespace disc_ral {
namespace {

// Arrange the insert-point of `op`'s operands in the same block. Do not deal
// with cross-block operands.
void ArrangeOperandsInsertPointInBlock(Operation* op) {
  for (auto operand : GetAllPossibleUsedValues(op)) {
    auto operandOp = operand.getDefiningOp();
    // Note that `isBeforeInBlock` also check whether `op` and `operandOp` are
    // in the same block.
    if ((operandOp != nullptr) && !operandOp->isBeforeInBlock(op)) {
      operandOp->moveBefore(op);
      ArrangeOperandsInsertPointInBlock(operandOp);
    }
  }
}

void BuildBlockGraphCycles(Block* block,
                           std::unique_ptr<GraphCycles>& cycle_detector,
                           DenseMap<Operation*, int64_t>& op_to_node_id) {
  std::vector<Operation*> op_list;
  op_to_node_id.clear();
  for (Operation& op : *block) {
    op_to_node_id.try_emplace(&op, op_list.size());
    op_list.push_back(&op);
  }
  cycle_detector.reset(new GraphCycles(op_list.size()));
  for (int64_t node_id = 0; node_id < op_list.size(); node_id++) {
    Operation* op = op_list[node_id];
    for (Value operand : GetAllPossibleUsedValues(op)) {
      Operation* operand_op = operand.getDefiningOp();
      // Only consider the operand_op inside the target block.
      auto iter = op_to_node_id.find(operand_op);
      if (iter == op_to_node_id.end()) {
        continue;
      }
      cycle_detector->InsertEdge(iter->second, node_id);
    }
  }
}

// 一个cluster存储了一个dot的集合
// 并且有显示的cluster_id
struct DotCluster {
  DotCluster(Operation* op, int op_id) : leader_op_id(op_id) {
    ops.push_back(op);
  }

  // Merges `other` into this cluster, and clears `other`.
  void merge(DotCluster& other) {
    // 经典SmallVector merge手法
    ops.insert(ops.end(), other.ops.begin(), other.ops.end());
    other.ops.clear();
  }

  // ID of the representative node of this cluster.
  int leader_op_id;

  // Dot ops to be batched.
  SmallVector<Operation*> ops;
};

// The `src` cluster will be cleared if the merge is succeed.
// 尝试将src cluster merge入到dst cluster中
void TryMergeDotClusters(DotCluster& dst, DotCluster& src,
                         std::unique_ptr<GraphCycles>& cycle_detector) {
  // Check cycle.
  int64_t dst_id = dst.leader_op_id;
  int64_t src_id = src.leader_op_id;
  // 判断是否可以merge，重点是判断是否有cycle的产生。
  auto optional_merged_id = TryMergeNode(cycle_detector.get(), dst_id, src_id);
  if (!optional_merged_id.has_value()) {
    // It forms a cycle.
    return;
  }
  // 通过判断没有cycle产生，进行merge操作。
  dst.merge(src);
  dst.leader_op_id = *optional_merged_id;
}

// The `MergingShapeEqualityMap` is a map whose value is a list of dots with
// same `MergingShape`.
// 由于有两种策略，所以选择使用template function
template <typename MergingShapeEqualityMap>
void BuildDotClusters(Block* block,
                      const MergingShapeEqualityMap& equal_merge_shape_map,
                      SmallVector<DotCluster>& merging_clusters) {
  merging_clusters.clear();
  // Dot ops in `equal_merge_shape_map` can be merged together if the merging
  // does not introduce cycle.

  // Form cycle detector.
  // 构造cycle监视器，如果会引入cycle，则不允许merge
  std::unique_ptr<GraphCycles> cycle_detector(new GraphCycles(0));
  DenseMap<Operation*, int64_t> op_to_node_id;
  BuildBlockGraphCycles(block, cycle_detector, op_to_node_id);

  // Find merge clusters.
  // 针对每一个dot的集合，尝试找寻哪些op可以构建成一个dot
  for (auto& dots_can_merge : equal_merge_shape_map) {
    auto ops = dots_can_merge.second;
    // 如果这个集合的大小小于2，则不需要建立dot集和
    if (ops.size() < 2) {
      continue;
    }
    SmallVector<DotCluster> clusters;
    for (auto op : ops) {
      // 初始每个op一个cluster
      clusters.emplace_back(op.getOperation(),
                            op_to_node_id[op.getOperation()]);
    }
    for (int64_t i = 0; i < clusters.size(); i++) {
      auto& merged = clusters[i];
      if (merged.ops.empty()) {
        continue;
      }
      for (int64_t j = i + 1; j < clusters.size(); j++) {
        // 尝试对于op之后的operation，尝试一点一点的merge起来
        // 这里时间复杂度是O(n^2)，有点高
        auto& to_merge = clusters[j];
        if (to_merge.ops.empty()) {
          continue;
        }
        // Try merge.
        // 这里的所有op，是先前通过map收集的在Left或是Right模式中shape一致的op，即潜在的融合op
        TryMergeDotClusters(merged, to_merge, cycle_detector);
      }
    }

    // 最后更新一遍cluster
    for (auto& cluster : clusters) {
      if (cluster.ops.size() > 1) {
        merging_clusters.push_back(cluster);
      }
    }
  }
}

class DotShareOperandMergeConverter {
 public:
  DotShareOperandMergeConverter(func::FuncOp func) : func_(func){};
  void run();

 public:
  enum ShareType : int32_t { LEFT = 0, RIGHT = 1 };

  // 一个DotShareInfo包含了一个dot的共享操作数和维度信息
  struct DotShareInfo {
    Value share_operand;
    mhlo::DotDimensionNumbersAttr dimension_numbers;

    // 重载一个==运算符
    bool operator==(const DotShareInfo& other) const {
      return (share_operand == other.share_operand) &&
             (dimension_numbers == other.dimension_numbers);
    }
  };

  struct DotShareInfoHash {
    std::size_t operator()(const DotShareInfo& dotShareInfo) const {
      std::size_t hash = mlir::hash_value(dotShareInfo.share_operand);
      const auto& dimension_numbers = dotShareInfo.dimension_numbers;
      const auto& lhs_batch_dims = dimension_numbers.getLhsBatchingDimensions();
      const auto& rhs_batch_dims = dimension_numbers.getRhsBatchingDimensions();
      const auto& lhs_contracting_dims =
          dimension_numbers.getLhsContractingDimensions();
      const auto& rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(lhs_batch_dims.begin(),
                                                         lhs_batch_dims.end()));
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(rhs_batch_dims.begin(),
                                                         rhs_batch_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(lhs_contracting_dims.begin(),
                                         lhs_contracting_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(rhs_contracting_dims.begin(),
                                         rhs_contracting_dims.end()));
      return hash;
    }
  };

  // 定义一个无序map，key是一个DotShareInfo，value是一个dot集合，DotShareInfoHash是用于生成hash键值
  using ShareOperandMap =
      std::unordered_map<DotShareInfo, SmallVector<mhlo::DotGeneralOp>,
                         DotShareInfoHash>;

 private:
  bool buildShareOperandMap(Block* block, ShareOperandMap& share_operand_map,
                            ShareType share_type);
  bool applyMerging(DotCluster& cluster, ShareType share_type);

 private:
  func::FuncOp func_;
};

// 合并两个operation，有相同的operand
// 比如dot(x,y) 和 dot(x,z) 可以合并成 dot(concat(x,y),z)
void DotShareOperandMergeConverter::run() {
  SmallVector<Block*> blocks;
  func_.walk([&](Block* block) { blocks.push_back(block); });
  for (Block* block : blocks) {
    // 针对的是二元操作，所以遍历是left相同还是right相同
    // todo：感觉这里有可扩展的地方
    for (auto share_type : SmallVector<ShareType, 2>({LEFT, RIGHT})) {
      // 显示使用一个map，来存储一个dot集群映射，该dot集群需要有相同的shape和维度数据
      // A map to help to cluster dots with same shape and dim-numbers together.
      ShareOperandMap share_operand_map;
      // 构建share_operand_map
      if (!buildShareOperandMap(block, share_operand_map, share_type)) {
        continue;
      }
      // Find merging clusters.
      // 根据我们的map，去做dot的merge操作
      SmallVector<DotCluster> merging_clusters;
      // 根据shareOperandMap来构建初始dot集群
      // 使用template模版方法
      BuildDotClusters<ShareOperandMap>(block, share_operand_map,
                                        merging_clusters);
      // Apply merging.
      // 对于明确可以merge的dot集群，进行merge操作
      for (auto& cluster : merging_clusters) {
        applyMerging(cluster, share_type);
      }
    }
  }
}

// 针对给定block，按照share_type来构建一个share_operand_map
bool DotShareOperandMergeConverter::buildShareOperandMap(
    Block* block, ShareOperandMap& share_operand_map, ShareType share_type) {
  // 遍历block里的的所有dot general op
  block->walk([&](mhlo::DotGeneralOp op) {
    // get one-side operand shareinfo according to the share_type
    DotShareInfo share_info;
    // 按照share_tye收集每个op的share_operand以及维度信息
    share_info.share_operand = (share_type == LEFT) ? op.getLhs() : op.getRhs();
    share_info.dimension_numbers = op.getDotDimensionNumbers();
    // 添加对应的op作为value，shareinfo为键
    auto& shared_op_list = share_operand_map[std::move(share_info)];
    shared_op_list.push_back(op);
  });
  return true;
}

// 完成实际的merge操作
bool DotShareOperandMergeConverter::applyMerging(DotCluster& cluster,
                                                 ShareType share_type) {
  auto& ops = cluster.ops;
  auto loc = ops.front()->getLoc();
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  // Move all dot ops, and their consumers if necessary, before the original
  // foremost dot. This makes sure that the newly created ops in this function
  // dominates their uses.
  for (auto op : ops) {
    if (foremost == op) {
      continue;
    }
    op->moveBefore(foremost);
    ArrangeOperandsInsertPointInBlock(op);
  }

  auto foremost_dot = dyn_cast<mhlo::DotGeneralOp>(foremost);

  OpBuilder builder(foremost_dot);
  auto non_concat_op =
      (share_type == LEFT) ? foremost_dot.getLhs() : foremost_dot.getRhs();
  auto orig_lhs_type =
      foremost_dot.getLhs().getType().dyn_cast<RankedTensorType>();
  auto orig_rhs_type =
      foremost_dot.getRhs().getType().dyn_cast<RankedTensorType>();
  auto orig_concat_op_type =
      (share_type == LEFT) ? orig_rhs_type : orig_lhs_type;
  auto orig_non_concat_op_type =
      (share_type == LEFT) ? orig_lhs_type : orig_rhs_type;
  auto orig_result_type = foremost_dot.getType().dyn_cast<RankedTensorType>();

  // All op share the same rank and elementType.
  auto rank = orig_non_concat_op_type.getRank();
  auto element_type = orig_non_concat_op_type.getElementType();

  // Find concat dim.
  int64_t concat_dim = -1;
  DenseSet<int64_t> non_concat_dims;
  auto dim_numbers = foremost_dot.getDotDimensionNumbers();
  // Batching and contracting dims of the to-concat operands.
  const auto& batch_dims = (share_type == LEFT)
                               ? dim_numbers.getRhsBatchingDimensions()
                               : dim_numbers.getLhsBatchingDimensions();
  const auto& contract_dims = (share_type == LEFT)
                                  ? dim_numbers.getRhsContractingDimensions()
                                  : dim_numbers.getLhsContractingDimensions();
  non_concat_dims.insert(batch_dims.begin(), batch_dims.end());
  non_concat_dims.insert(contract_dims.begin(), contract_dims.end());

  assert(rank == non_concat_dims.size() + 1);
  for (int64_t i = 0; i < rank; i++) {
    if (!non_concat_dims.contains(i)) {
      concat_dim = i;
      break;
    }
  }
  assert(concat_dim > 0);

  // Initial to_concat_ops, concat_dim_sum, is_dynamic_shape.
  SmallVector<Value> to_concat_ops;
  bool is_dynamic_shape = false;
  int64_t concat_dim_sum = 0;
  for (auto op : ops) {
    mhlo::DotGeneralOp dot = dyn_cast<mhlo::DotGeneralOp>(op);
    auto to_concat = (share_type == LEFT) ? dot.getRhs() : dot.getLhs();
    auto concat_op_type = to_concat.getType().dyn_cast<RankedTensorType>();
    auto concat_dim_size = concat_op_type.getDimSize(concat_dim);
    if (concat_dim_size == ShapedType::kDynamic) {
      is_dynamic_shape = true;
      concat_dim_sum = ShapedType::kDynamic;
    } else if (!is_dynamic_shape) {
      concat_dim_sum += concat_dim_size;
    }
    to_concat_ops.push_back(to_concat);
  }

  // Build concat op.
  SmallVector<int64_t, 4> concat_op_shapes(rank, ShapedType::kDynamic);
  for (int64_t i = 0; i < rank; i++) {
    if (i != concat_dim) {
      concat_op_shapes[i] = orig_concat_op_type.getDimSize(i);
    } else if (!is_dynamic_shape) {
      concat_op_shapes[i] = concat_dim_sum;
    }
  }
  auto concat_op_type = RankedTensorType::get(concat_op_shapes, element_type);
  auto concat_op = builder.create<mhlo::ConcatenateOp>(
      loc, concat_op_type, to_concat_ops, concat_dim);

  // According to `DotGeneralOp::reifyReturnTypeShapes`, m and n dims are the
  // last two dimensions. In other words, the result rank order of the
  // dot_general is always (batch,m,n). So the concat_dim may change, and we
  // need to use concat_dim_in_result other than concat_dim.
  auto concat_dim_in_result = (share_type == LEFT) ? (rank - 1) : (rank - 2);

  // Build result type.
  SmallVector<int64_t, 4> result_shapes(rank, ShapedType::kDynamic);
  for (int64_t i = 0; i < rank; i++) {
    if (i != concat_dim_in_result) {
      result_shapes[i] = orig_result_type.getDimSize(i);
    } else if (!is_dynamic_shape) {
      result_shapes[i] = concat_dim_sum;
    }
  }
  result_shapes[concat_dim_in_result] = concat_dim_sum;
  auto result_type = RankedTensorType::get(result_shapes, element_type);
  auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), dim_numbers.getLhsBatchingDimensions(),
      dim_numbers.getRhsBatchingDimensions(),
      dim_numbers.getLhsContractingDimensions(),
      dim_numbers.getRhsContractingDimensions());

  // Build result DotGeneralOp
  auto lhs = (share_type == LEFT) ? non_concat_op : concat_op.getResult();
  auto rhs = (share_type == LEFT) ? concat_op.getResult() : non_concat_op;
  Value merged_dot = builder.create<mhlo::DotGeneralOp>(
      loc, result_type, lhs, rhs, dot_dimension_attr, nullptr);

  // Build slice for each of the original dot op, and replace the dot.
  if (result_type.getNumDynamicDims() == 0) {
    // Use static-dim ops.
    int64_t concat_dim_start = 0;
    for (int64_t i = 0; i < ops.size(); i++) {
      mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
      auto orig_dot_type = op.getType().dyn_cast<RankedTensorType>();

      SmallVector<int64_t> start(rank, 0);
      SmallVector<int64_t> limit(rank);
      SmallVector<int64_t> strides(rank, 1);
      start[concat_dim_in_result] = concat_dim_start;
      for (int64_t j = 0; j < rank; j++) {
        if (j == concat_dim_in_result) {
          limit[j] =
              concat_dim_start + orig_dot_type.getDimSize(concat_dim_in_result);
          concat_dim_start = limit[j];
        } else {
          limit[j] = orig_dot_type.getDimSize(j);
        }
      }
      Value slice = builder.create<mhlo::SliceOp>(
          loc, merged_dot, GetI64ElementsAttr(start, &builder),
          GetI64ElementsAttr(limit, &builder),
          GetI64ElementsAttr(strides, &builder));

      auto cast_slice = builder.create<tensor::CastOp>(
          loc, op->getResult(0).getType(), slice);
      op->replaceAllUsesWith(cast_slice);
    }
  } else {
    // Use dynamic-dim ops.
    Value concat_dim_start = builder.create<arith::ConstantIndexOp>(loc, 0);
    Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
    Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    for (int64_t i = 0; i < ops.size(); i++) {
      mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
      // Note that we do not use op but op's rhs and lhs to build DimOp.
      // Because op will be replaced and if use op to build DimOp, it will
      // form circles.
      auto concatenate_op = (share_type == LEFT) ? op.getRhs() : op.getLhs();
      auto non_concatenate_op =
          (share_type == LEFT) ? op.getLhs() : op.getRhs();
      auto orig_dot_type = op.getType().dyn_cast<RankedTensorType>();

      SmallVector<Value, 4> start_values(rank, zero);
      SmallVector<Value, 4> limit_values(rank);
      SmallVector<Value, 4> strides_values(rank, one);

      for (int64_t j = 0; j < rank; j++) {
        if (j == concat_dim_in_result) {
          start_values[j] = concat_dim_start;
          limit_values[j] = builder.create<arith::AddIOp>(
              loc, start_values[j],
              builder.create<tensor::DimOp>(loc, concatenate_op,
                                            concat_dim_in_result));
          concat_dim_start = limit_values[j];
        } else {
          limit_values[j] =
              builder.create<tensor::DimOp>(loc, non_concatenate_op, j);
        }
      }

      auto index_ty = builder.getIndexType();
      auto start_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(start_values.size())},
                                index_ty),
          start_values);
      auto limit_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(limit_values.size())},
                                index_ty),
          limit_values);
      auto strides_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides_values.size())},
                                index_ty),
          strides_values);
      SmallVector<int64_t, 4> slice_shapes(rank, ShapedType::kDynamic);
      for (int64_t j = 0; j < rank; j++) {
        slice_shapes[j] = orig_dot_type.getDimSize(j);
      }
      auto slice_type = RankedTensorType::get(slice_shapes, element_type);

      Value dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, merged_dot, start_indices, limit_indices,
          strides_indices);
      auto cast_dyn_slice = builder.create<tensor::CastOp>(
          loc, op->getResult(0).getType(), dyn_slice);

      op->replaceAllUsesWith(cast_dyn_slice);
    }
  }

  // No longer need the original dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }

  return true;
}

// 批次融合，从而显示添加原批次维度
// 如下是一个维度拼接例子：
// OperandShape Merge 维度拼接示例
// 合并前形状: [B, K1, M] 和 [B, K2, M]  
// 合并后形状: [B, K1+K2, M]
class DotBatchMergeConverter {
 public:
  DotBatchMergeConverter(func::FuncOp func) : func_(func){};
  bool run();

 public:
  struct MergingShape {
    // TODO: use shape-symbol rather than dim-value of m/n/k in this struct
    // after reconstructing ShapeAnalysis. We actually do not need to
    // distinguish m/n/k but only need to use the overall shape-symbol here in
    // the future.
    DimValue m_dim;
    DimValue n_dim;
    DimValue contracting_dim;
    SmallVector<DimValue> batching_dims;
    mhlo::DotDimensionNumbersAttr dimension_numbers;
    Type element_type;
    bool operator==(const MergingShape& other) const {
      return (m_dim == other.m_dim) && (n_dim == other.n_dim) &&
             (contracting_dim == other.contracting_dim) &&
             (batching_dims == other.batching_dims) &&
             (dimension_numbers == other.dimension_numbers) &&
             (element_type == other.element_type);
    }
  };

  struct MergingShapeHash {
    std::size_t operator()(const MergingShape& dotShape) const {
      std::size_t hash = dotShape.m_dim.hash();
      hash = llvm::hash_combine(hash, dotShape.n_dim.hash());
      hash = llvm::hash_combine(hash, dotShape.contracting_dim.hash());
      for (const auto& d : dotShape.batching_dims) {
        hash = llvm::hash_combine(hash, d.hash());
      }
      const auto& dimension_numbers = dotShape.dimension_numbers;
      const auto& lhs_batch_dims = dimension_numbers.getLhsBatchingDimensions();
      const auto& rhs_batch_dims = dimension_numbers.getRhsBatchingDimensions();
      const auto& lhs_contracting_dims =
          dimension_numbers.getLhsContractingDimensions();
      const auto& rhs_contracting_dims =
          dimension_numbers.getRhsContractingDimensions();
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(lhs_batch_dims.begin(),
                                                         lhs_batch_dims.end()));
      hash = llvm::hash_combine(hash,
                                llvm::hash_combine_range(rhs_batch_dims.begin(),
                                                         rhs_batch_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(lhs_contracting_dims.begin(),
                                         lhs_contracting_dims.end()));
      hash = llvm::hash_combine(
          hash, llvm::hash_combine_range(rhs_contracting_dims.begin(),
                                         rhs_contracting_dims.end()));
      hash = llvm::hash_combine(hash, mlir::hash_value(dotShape.element_type));
      return hash;
    }
  };

  using MergingShapeEqualMap =
      std::unordered_map<MergingShape, SmallVector<mhlo::DotGeneralOp>,
                         MergingShapeHash>;

 private:
  bool buildMergingShapeMap(Block* block,
                            ShapeAnalysisDeprecated& analysisDeprecated,
                            MergingShapeEqualMap& equal_merge_shape_map);
  bool applyMerging(DotCluster& cluster);
  Value expandDim0(OpBuilder& builder, Location& loc, Value value);

 private:
  func::FuncOp func_;
};

// 做batch维度融合
bool DotBatchMergeConverter::run() {
  // 形状分析：建立张量维度符号系统
  ShapeAnalysisDeprecated analysisDeprecated(func_);
  if (failed(analysisDeprecated.run())) {
    LLVM_DEBUG(llvm::dbgs()
               << "ShapeAnalysisDeprecated failes for dot merge.\n");
    return false;
  }

  // 遍历函数的每一个block
  // 可以发现，其代码逻辑和DotShareOperandMergeConverter是类似的
  // 核心就在ShapeAnalysisDeprecated分析和buildMergingShapeMap，均是个性化的函数设计
  SmallVector<Block*> blocks;
  func_.walk([&](Block* block) { blocks.push_back(block); });

  for (Block* block : blocks) {
    // A map to help to cluster dots with same shape and dim-numbers together.
    MergingShapeEqualMap equal_merge_shape_map;
    if (!buildMergingShapeMap(block, analysisDeprecated,
                              equal_merge_shape_map)) {
      continue;
    }
    // Find merging clusters.
    SmallVector<DotCluster> merging_clusters;
    BuildDotClusters<MergingShapeEqualMap>(block, equal_merge_shape_map,
                                           merging_clusters);
    // Apply merging.
    for (auto& cluster : merging_clusters) {
      applyMerging(cluster);
    }
  }

  return true;
}

// 形状聚类映射构建
bool DotBatchMergeConverter::buildMergingShapeMap(
    Block* block, ShapeAnalysisDeprecated& analysisDeprecated,
    MergingShapeEqualMap& equal_merge_shape_map) {
  block->walk([&](mhlo::DotGeneralOp op) {
    MergingShape dot_shape;
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    // Initialize `dimension_numbers`.
    // 提取维度配置信息
    dot_shape.dimension_numbers = op.getDotDimensionNumbers();
    // 分析批次维度符号
    auto lhs_batch_dims =
        dot_shape.dimension_numbers.getLhsBatchingDimensions();
    auto rhs_batch_dims =
        dot_shape.dimension_numbers.getRhsBatchingDimensions();
    assert(lhs_batch_dims.size() == rhs_batch_dims.size());
    // Initialize `batching_dims`.
    SmallVector<DimValue>& batching_dims = dot_shape.batching_dims;
    for (auto dim : lhs_batch_dims) {
      batching_dims.emplace_back(
          std::move(analysisDeprecated.getDimValue(lhs, dim)));
    }
    // Initialize `contracting_dim`.
    // 分析收缩维度符号
    auto lhs_contracting_dims =
        dot_shape.dimension_numbers.getLhsContractingDimensions();
    assert(lhs_contracting_dims.size() == 1);
    dot_shape.contracting_dim =
        analysisDeprecated.getDimValue(lhs, lhs_contracting_dims[0]);
    // Initialize `m_dim`.
    int64_t lhs_rank = lhs.getType().cast<RankedTensorType>().getRank();
    assert(lhs_batch_dims.size() + 2 == lhs_rank);
    DenseSet<int64_t> lhs_batch_dims_set(lhs_batch_dims.begin(),
                                         lhs_batch_dims.end());
    for (int64_t i = 0; i < lhs_rank; i++) {
      if ((lhs_batch_dims_set.find(i) == lhs_batch_dims_set.end()) &&
          (i != lhs_contracting_dims[0])) {
        dot_shape.m_dim = analysisDeprecated.getDimValue(lhs, i);
        break;
      }
    }
    // Initialize `n_dim`.
    int64_t rhs_rank = rhs.getType().cast<RankedTensorType>().getRank();
    assert(rhs_batch_dims.size() + 2 == rhs_rank);
    DenseSet<int64_t> rhs_batch_dims_set(rhs_batch_dims.begin(),
                                         rhs_batch_dims.end());
    auto rhs_contracting_dims =
        dot_shape.dimension_numbers.getRhsContractingDimensions();
    assert(rhs_contracting_dims.size() == 1);
    for (int64_t i = 0; i < rhs_rank; i++) {
      if ((rhs_batch_dims_set.find(i) == rhs_batch_dims_set.end()) &&
          (i != rhs_contracting_dims[0])) {
        dot_shape.n_dim = analysisDeprecated.getDimValue(rhs, i);
        break;
      }
    }
    // Initialize `element_type`.
    dot_shape.element_type =
        op.getType().cast<RankedTensorType>().getElementType();

    auto& op_list = equal_merge_shape_map[std::move(dot_shape)];
    op_list.push_back(op);
  });
  return true;
}

bool DotBatchMergeConverter::applyMerging(DotCluster& cluster) {
  auto& ops = cluster.ops;
  auto loc = ops.front()->getLoc();
  auto foremost = ops.front();
  for (int64_t i = 1; i < ops.size(); i++) {
    auto& op = ops[i];
    if (op->isBeforeInBlock(foremost)) {
      foremost = op;
    }
  }
  // Move all dot ops, and their consumers if necessary, before the original
  // foremost dot. This makes sure that the newly created ops in this function
  // dominates their uses.
  for (auto op : ops) {
    if (foremost == op) {
      continue;
    }
    op->moveBefore(foremost);
    ArrangeOperandsInsertPointInBlock(op);
  }
  auto last_dot = dyn_cast<mhlo::DotGeneralOp>(foremost);
  // We use the foremost dot to create the builder. Thus we only need to reorder
  // the operands of some newly created ops, rather users of them.
  OpBuilder builder(last_dot);
  auto orig_lhs_type = last_dot.getLhs().getType().dyn_cast<RankedTensorType>();
  auto orig_rhs_type = last_dot.getRhs().getType().dyn_cast<RankedTensorType>();
  auto orig_dot_type = last_dot.getType().dyn_cast<RankedTensorType>();
  SmallVector<Value, 4> lhs_operands;
  SmallVector<Value, 4> rhs_operands;
  for (auto op : ops) {
    mhlo::DotGeneralOp dot = dyn_cast<mhlo::DotGeneralOp>(op);
    auto lhs_expand = expandDim0(builder, loc, dot.getLhs());
    auto rhs_expand = expandDim0(builder, loc, dot.getRhs());
    if (!lhs_expand || !rhs_expand) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to expand dim for dot merge.\n");
      return false;
    }
    lhs_operands.push_back(lhs_expand);
    rhs_operands.push_back(rhs_expand);
  }
  auto concat_dim = builder.getI64IntegerAttr(0);
  // Concat lhs.
  auto lhs_rank = orig_lhs_type.getRank() + 1;
  SmallVector<int64_t, 4> lhs_shapes(lhs_rank, ShapedType::kDynamic);
  lhs_shapes[0] = ops.size();
  for (int64_t i = 1; i < lhs_rank; i++) {
    lhs_shapes[i] = orig_lhs_type.getDimSize(i - 1);
  }
  auto lhs_type =
      RankedTensorType::get(lhs_shapes, orig_lhs_type.getElementType());
  Value lhs = builder.create<mhlo::ConcatenateOp>(loc, lhs_type, lhs_operands,
                                                  concat_dim);
  // Concat rhs.
  auto rhs_rank = orig_rhs_type.getRank() + 1;
  SmallVector<int64_t, 4> rhs_shapes(rhs_rank, ShapedType::kDynamic);
  rhs_shapes[0] = ops.size();
  for (int64_t i = 1; i < rhs_rank; i++) {
    rhs_shapes[i] = orig_rhs_type.getDimSize(i - 1);
  }
  auto rhs_type =
      RankedTensorType::get(rhs_shapes, orig_rhs_type.getElementType());
  Value rhs = builder.create<mhlo::ConcatenateOp>(loc, rhs_type, rhs_operands,
                                                  concat_dim);
  // Result type.
  auto result_rank = orig_dot_type.getRank() + 1;
  SmallVector<int64_t, 4> result_shapes(result_rank, ShapedType::kDynamic);
  result_shapes[0] = ops.size();
  for (int64_t i = 1; i < result_rank; i++) {
    result_shapes[i] = orig_dot_type.getDimSize(i - 1);
  }
  auto result_type =
      RankedTensorType::get(result_shapes, orig_dot_type.getElementType());
  // Build dot dimension numbers.
  auto dim_numbers = last_dot.getDotDimensionNumbers();

  SmallVector<int64_t> lhs_batching_dims;
  auto lhs_batch = dim_numbers.getLhsBatchingDimensions();
  lhs_batching_dims.push_back(0);
  for (auto& val : lhs_batch) {
    lhs_batching_dims.push_back(val + 1);
  }
  SmallVector<int64_t> rhs_batching_dims;
  auto rhs_batch = dim_numbers.getRhsBatchingDimensions();
  rhs_batching_dims.push_back(0);
  for (auto& val : rhs_batch) {
    rhs_batching_dims.push_back(val + 1);
  }

  SmallVector<int64_t> lhs_contracting_dims;
  auto lhs_contract = dim_numbers.getLhsContractingDimensions();
  for (auto& val : lhs_contract) {
    lhs_contracting_dims.push_back(val + 1);
  }

  SmallVector<int64_t> rhs_contracting_dims;
  auto rhs_contract = dim_numbers.getRhsContractingDimensions();
  for (auto& val : rhs_contract) {
    rhs_contracting_dims.push_back(val + 1);
  }

  auto dot_dimension_attr = mhlo::DotDimensionNumbersAttr::get(
      builder.getContext(), lhs_batching_dims, rhs_batching_dims,
      lhs_contracting_dims, rhs_contracting_dims);

  // Create batched dot.
  Value batched_dot = builder.create<mhlo::DotGeneralOp>(
      loc, result_type, lhs, rhs, dot_dimension_attr, nullptr);

  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  // Build slice and reshape op for each of the original dot op, and replace the
  // dot.
  for (int64_t i = 0; i < ops.size(); i++) {
    mhlo::DotGeneralOp op = dyn_cast<mhlo::DotGeneralOp>(ops[i]);
    if (orig_dot_type.getNumDynamicDims() == 0) {
      // Use static-dim ops.
      SmallVector<int64_t> start(result_rank, 0);
      start[0] = i;
      SmallVector<int64_t> limit(result_rank);
      limit[0] = i + 1;
      for (int64_t i = 1; i < result_rank; i++) {
        limit[i] = orig_dot_type.getDimSize(i - 1);
      }
      SmallVector<int64_t> strides(result_rank, 1);
      auto slice = builder.create<mhlo::SliceOp>(
          loc, batched_dot, GetI64ElementsAttr(start, &builder),
          GetI64ElementsAttr(limit, &builder),
          GetI64ElementsAttr(strides, &builder));
      auto reshape = builder.create<mhlo::ReshapeOp>(loc, orig_dot_type, slice);
      op->replaceAllUsesWith(reshape);
    } else {
      // Use dynamic-dim ops.
      SmallVector<Value, 4> strides_values(result_rank, one);
      SmallVector<Value, 4> start_values(result_rank, zero);
      start_values[0] = builder.create<arith::ConstantIndexOp>(loc, i);
      SmallVector<Value, 4> limit_values;
      limit_values.push_back(
          builder.create<arith::ConstantIndexOp>(loc, i + 1));
      for (int64_t j = 1; j < result_rank; j++) {
        limit_values.push_back(
            builder.create<tensor::DimOp>(loc, batched_dot, j));
      }
      auto index_ty = builder.getIndexType();
      auto start_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(start_values.size())},
                                index_ty),
          start_values);
      auto limit_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(limit_values.size())},
                                index_ty),
          limit_values);
      auto strides_indices = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get({static_cast<int64_t>(strides_values.size())},
                                index_ty),
          strides_values);
      SmallVector<int64_t, 4> slice_shapes(result_rank, ShapedType::kDynamic);
      slice_shapes[0] = 1;
      for (int64_t i = 1; i < result_rank; i++) {
        slice_shapes[i] = orig_dot_type.getDimSize(i - 1);
      }
      auto slice_type =
          RankedTensorType::get(slice_shapes, orig_dot_type.getElementType());
      auto dyn_slice = builder.create<mhlo::RealDynamicSliceOp>(
          loc, slice_type, batched_dot, start_indices, limit_indices,
          strides_indices);
      SmallVector<Value, 4> reshape_shape_values;
      for (int64_t j = 1; j < slice_type.getRank(); j++) {
        reshape_shape_values.push_back(
            builder.create<tensor::DimOp>(loc, dyn_slice, j));
      }
      auto reshape_shape = builder.create<tensor::FromElementsOp>(
          loc,
          RankedTensorType::get(
              {static_cast<int64_t>(reshape_shape_values.size())}, index_ty),
          reshape_shape_values);
      auto dyn_reshape = builder.create<mhlo::DynamicReshapeOp>(
          loc, orig_dot_type, dyn_slice, reshape_shape);
      op->replaceAllUsesWith(dyn_reshape);
    }
  }

  // No longer need the original dot ops.
  for (int64_t i = 0; i < ops.size(); i++) {
    ops[i]->erase();
  }

  return true;
}

// Expand dim 0. We use reshape op to expand it currently. We plan to use
// tensor.ExpandOp in the future.
Value DotBatchMergeConverter::expandDim0(OpBuilder& builder, Location& loc,
                                         Value value) {
  auto type = value.getType().dyn_cast<RankedTensorType>();
  if (!type) {
    return nullptr;
  }
  SmallVector<int64_t, 4> result_dims;
  result_dims.push_back(1);
  bool is_static = true;
  for (int64_t i = 0; i < type.getRank(); i++) {
    int64_t dim = type.getDimSize(i);
    is_static &= (dim != ShapedType::kDynamic);
    result_dims.push_back(dim);
  }
  auto result_type = RankedTensorType::get(result_dims, type.getElementType());
  if (is_static) {
    return builder.create<mhlo::ReshapeOp>(loc, result_type, value);
  }
  SmallVector<Value, 4> dims;
  dims.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));
  for (int64_t i = 0; i < type.getRank(); i++) {
    dims.push_back(builder.create<tensor::DimOp>(loc, value, i));
  }
  auto result_shape = builder.create<tensor::FromElementsOp>(
      loc,
      RankedTensorType::get({static_cast<int64_t>(dims.size())},
                            builder.getIndexType()),
      dims);
  auto dyn_reshape = builder.create<mhlo::DynamicReshapeOp>(
      loc, result_type, value, result_shape);
  return dyn_reshape;
}

// 将两个相关的dot ops合并成一个dot op
struct DiscDotMergePass : public DiscDotMergePassBase<DiscDotMergePass> {
  DiscDotMergePass()
      : DiscDotMergePassBase<DiscDotMergePass>::DiscDotMergePassBase() {}
  void runOnOperation() override;

 private:
  void dotShareOperandMerging(func::FuncOp& func);
  bool dotBatchMerging(func::FuncOp& func);
};

void DiscDotMergePass::runOnOperation() {
  func::FuncOp func = getOperation();
  dotShareOperandMerging(func);
  if (!dotBatchMerging(func)) {
    signalPassFailure();
  }
}

void DiscDotMergePass::dotShareOperandMerging(func::FuncOp& func) {
  DotShareOperandMergeConverter(func).run();
}

bool DiscDotMergePass::dotBatchMerging(func::FuncOp& func) {
  return DotBatchMergeConverter(func).run();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createDiscDotMergePass() {
  return std::make_unique<DiscDotMergePass>();
}

}  // namespace disc_ral
}  // namespace mlir
