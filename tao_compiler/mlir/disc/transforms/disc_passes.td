/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

include "mlir/Pass/PassBase.td"

def DiscToLLVMPass : Pass<"disc-to-llvm", "ModuleOp"> {
  let summary = "Pass for applying LLVM legalization patterns.";
  let constructor = "createDiscToLLVMPass()";
}

def SplitLargeOpsPass : Pass<"split-large-ops", "mlir::func::FuncOp"> {
  let summary = "Split large hlo ops (e.g. having too many operands) into a "
                "sequence of hlo ops.";
  let constructor = "createDiscSplitLargeOpsPass()";
  let options = [
    Option<"max_num_operands_per_op_", "max-num-operands-per-op", "int",
           /*default=*/"32", "Maximum allowed number of operands per op.">,
  ];
}

// 将static shape info从jit static rank compilation cache中提取出来，更新到IR中
// 在IR中用attribute形式
def ReviseArgumentsForStaticRankPass : Pass<"disc-tf-revise-args-for-static-rank", "ModuleOp"> {
  let summary = "extract the const/fixed-shape input info from jit static rank compilation cache, and update the info in the IR.";
  let constructor = "createReviseArgsForStaticRankPass()";
}

def DiscLhloLegalizeRootsToParallelLoopsPass : Pass<"disc-lhlo-legalize-roots-to-parallel-loops", "mlir::func::FuncOp"> {
  let summary = "Legalize the roots of lmhlo.fusion to parallel loops.";
  let constructor = "createDiscLhloLegalizeRootsToParallelLoopsPass()";
  let options = [
    Option<"core_count_", "core-count", "int",
            /*default=*/"-1", "core count (e.g., SM count on NVIDIA GPU).">,
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::math::MathDialect",
    "gpu::GPUDialect",
    "disc_shape::DISCShapeDialect"
  ];
}

def ConvRewriterPass : Pass<"disc-conv-rewriter", "mlir::func::FuncOp"> {
  let summary = "Canonicalize conv ops to be suitable for lowering to cudnn lib calls.";
  let constructor = "createDiscConvRewriter()";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
  ];
}

def DiscCollectiveOpsRewriterPass : Pass<"disc-collective-ops-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite mhlo collective ops to DISC custom library call";
  let constructor = "createDiscCollectiveOpsRewriterPass()";
}

def MhloDecompositionRewriterPass : Pass<"disc-mhlo-decomp-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite and decompose mhlo ops.";
  let constructor = "createDiscMhloDecompositionRewriterPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
  ];
}

def DotRewriterPass : Pass<"disc-dot-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite dot to fold transpose.";
  let constructor = "createDiscDotRewriterPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
  ];
}

def ReductionRewriterPass : Pass<"disc-reduction-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite mhlo::ReduceOp";
  let constructor = "createDiscReductionRewriterPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
  ];
}

def QuantizedDotRewriterPass : Pass<"quantized-disc-dot-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite quantized dot to transpose weight(row2col).";
  let constructor = "createDiscQuantizedDotRewriter()";
}

def GpuConvPaddingLegalizationPass : Pass<"disc-gpu-conv-padding-legalization", "mlir::func::FuncOp"> {
  let summary = "Convert conv ops' padding value to be suitable for lowering to cudnn lib calls.";
  let constructor = "createDiscGpuConvPaddingLegalization()";
}

def DiscSpecializeFusionWithSpeculationPass : Pass<"disc-specialize-fusion-with-speculation", "mlir::func::FuncOp"> {
  let summary = "Pass for specializing a fusion op with some speculations.";
  let constructor = "createDiscSpecializeFusionWithSpeculationPass()";
  let options = [
    Option<"core_count_", "core-count", "int",
            /*default=*/"-1", "core count (e.g., SM count on NVIDIA GPU).">,
    Option<"max_threads_per_core_", "max-threads-per-core", "int",
            /*default=*/"-1", "max thread number that a core supports concurrently.">,
    Option<"max_threads_per_block_", "max-threads-per-block", "int",
            /*default=*/"-1", "max thread number per block.">,
  ];
}

def ElementTypeConverterPass : Pass<"disc-element-type-converter", "mlir::func::FuncOp"> {
  let summary = "Eliminates certain element types as the input or output of ops "
                "by inserting Convert ops.";
  let constructor = "createDiscElementTypeConverterPass()";
  let options = [
    Option<"enable_fp16_gemm_", "enable-fp16-gemm", "bool",
           /*default=*/"false", "Enable fp16 for GEMM or not.">,
    Option<"enable_fp16_conv_", "enable-fp16-conv", "bool",
           /*default=*/"false", "Enable fp16 for CONV or not.">,
    Option<"promote_fp16_sensitive_ops_to_f32_", "promote-fp16-sensitive-ops-to-f32", "bool",
           /*default=*/"false", "Promote some fp16 opts to fp32 if enabled.">
  ];
}

def ParallelLoopCollapsing : Pass<"disc-parallel-loop-collapsing", "mlir::func::FuncOp"> {
  let summary = "Collapse parallel loops";
  let constructor = "createDiscParallelLoopCollapsingPass()";
}

def SCFParallelLoopTiling : Pass<"disc-parallel-loop-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile parallel loops";
  let constructor = "createParallelLoopTilingPass()";
  let options = [
    ListOption<"tileSizes", "parallel-loop-tile-sizes", "int64_t",
               "Factors to tile parallel loops by",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"withInboundCheck", "with-inbound-check", "bool",
           /*default=*/"false",
           "Perform tiling with fixed upperbound with inbound check "
           "for internal loops">
  ];
  let dependentDialects = ["affine::AffineDialect"];
}

// XLA style fusion strategey
def DiscFusionPass : Pass<"disc-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse lmhlo ops to kLoop/kInput fusion patterns.";
  let constructor = "createDiscFusionPass()";
  let options = [
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
    Option<"fusion_strategy_", "fusion-strategy", "std::string",
           /*default=*/"\"base\"", "Name of the fusion strategy.">,
    Option<"mlir_compute_intensive_codegen_", "mlir-compute-intensive-codegen", "bool",
           /*default=*/"false", "whether using MLIR based compute intensive codegen.">,
  ];
}

def DiscFuseSplatConstPass: Pass<"disc-fuse-splat-const", "mlir::func::FuncOp"> {
  let summary = "Fuse splat constant to avoid memory access.";
  let constructor = "createDiscFuseSplatConstPass()";
}

def DiscMarkShapeCalculationPass : Pass<"disc-mhlo-mark-shape-calc", "ModuleOp"> {
  let summary = "Identify and mark with an attribute operations performing shape calculation";
  let constructor = "createDiscMarkShapeCalcOpPass()";
}

def DiscConstToRALPass : Pass<"disc-const-to-ral", "ModuleOp"> {
  let summary = "convert lhlo ConstOp to ral const.";
  let constructor = "createDiscConstToRALPass()";
  let options = [
    Option<"metadata_file_path_", "metadata-file-name", "std::string",
           /*default=*/"\"metadata.pbtxt\"", "Name of the metadata file.">,
  ];
}

def PlaceOpsPass : Pass<"mhlo-place-ops", "ModuleOp"> {
  let summary = "Place shape calculation Ops on Cpu";
  let constructor = "createPlacerPass()";
  let options = [
    Option<"on_gpu_", "on-gpu", "bool",
           /*default=*/"true", "Data calculation Ops are placed on GPU or not">,
  ];
}

def ReviseGpuKernelOutliningPass : Pass<"disc-revise-gpu-kernel-outlining", "ModuleOp"> {
  let summary = "Revise the kernel outlining by expanding the host memref into scalars";
  let constructor = "createReviseGpuKernelOutliningPass()";
}

def DiscLowerToLibraryCallPass : Pass<"disc-lower-to-library-call", "mlir::func::FuncOp"> {
  let summary = "Lower some specific ops to library calls (modeled by `disc_ral.launch` op).";
  let constructor = "createDiscLowerToLibraryCallPass()";
  let options = [
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
  ];
}

def DiscAssignMemorySpacePass : Pass<"disc-assign-memory-space", "ModuleOp"> {
  let summary = "Assign memory space for each memref.";
  let constructor = "createDiscAssignMemorySpacePass()";
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
  ];
  let dependentDialects = [
    "gpu::GPUDialect"
  ];
}

def GpuKernelToBlobPass : Pass<"disc-gpu-kernel-to-blob", "gpu::GPUModuleOp"> {
  let summary = "Pass to annotate GPU Module with its PTX";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"7", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"5", "gpu sm cc_minor.">,
    Option<"multi_cc_support_", "multi-cc-support", "bool",
            /*default=*/"false", "multiple gpu types support in AOT compilation.">,
    Option<"multi_cc_support_dbg_ptx_only_", "multi-cc-support-dbg-ptx-only", "bool",
            /*default=*/"false", "compile to ptx only, only valid with multi-cc-support.">,
    Option<"blob_annotation_", "blob-annotation", "std::string",
           /*default=*/"\"gpu.binary_blob\"", "Blob attribute name">,
  ];
  let constructor = "CreateDiscGpuKernelToBlobPass()";
}

def ConvertShapeToStandardPass : Pass<"disc-convert-shape-to-std", "mlir::func::FuncOp"> {
  let summary = "Pass to convert shape ops to std dialect";
  let constructor = "createDiscConvertShapeToStandardPass()";
}

def ConvertTensorToStandardPass : Pass<"disc-convert-tensor-to-std", "mlir::func::FuncOp"> {
  let summary = "Pass to convert tensor ops to std dialect";
  let constructor = "createDiscConvertTensorToStandardPass()";
}

def InputInlineFusionPass : Pass<"disc-input-inline-fusion", "mlir::func::FuncOp"> {
  let summary = "Fuse the rest of the lmhlo nodes into the parallel loops after the roots "
    "has been expanded into loops in LhloLegalizeRootsToParallelLoops";
  let constructor = "createDiscInputInlineFusionPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "memref::MemRefDialect",
    "disc_shape::DISCShapeDialect"
  ];
}

def RemoveShapeConstraintsPass : Pass<"disc-remove-shape-constraint", "mlir::func::FuncOp"> {
  let summary = "Pass to remove all shape constraint ops";
  let constructor = "createDiscRemoveShapeConstraintsPass()";
}

def ConvertHloToStandardPass : Pass<"disc-convert-hlo-to-std", "mlir::func::FuncOp"> {
  let summary = "Pass to convert some hlo ops to std dialect";
  let constructor = "createDiscConvertHloToStandardPass()";
  let dependentDialects = [
    "arith::ArithDialect"
  ];
}

def StdBufferizePass : Pass<"disc-std-bufferize", "mlir::func::FuncOp"> {
  let summary = "Pass to convert tensor std.constant ops to hlo dialect";
  let constructor = "createDiscStdBufferizePass()";
}

def DiscMemRefCSEPass : Pass<"disc-memref-cse", "mlir::func::FuncOp"> {
  let summary = "Perform CSE on memref.load";
  let constructor = "createDiscMemRefCSEPass()";
}

def DiscMemRefLoadStoreSimplifierPass : Pass<"disc-memref-load-store-simplifier", "mlir::func::FuncOp"> {
  let summary = "Perform load/store simplifier on memref";
  let constructor = "createDiscMemRefLoadStoreSimplifierPass()";
}

def DiscLowerTfPass : Pass<"disc-lower-tf", "mlir::func::FuncOp"> {
  let summary = "Lowering some tensorflow ops";
  let constructor = "createDiscLowerTfPass()";
  let options = [
    Option<"pdll_files_", "pdll-files", "std::string",
           /*default=*/"\"\"", "comma seperated pathes to user defined pdll files">,
    Option<"pdll_include_dirs_", "pdll-include-dirs", "std::string",
           /*default=*/"\"\"", "comma seperated pathes to pdll include dirs">,
  ];
}

def RemoveDeadBufferPass : Pass<"disc-remove-dead-buffer", "mlir::func::FuncOp"> {
  let summary = "Pass to remove buffers that are not accessed by others";
  let constructor = "createDiscRemoveDeadBufferPass()";
}

def AssignKernelNamePass : Pass<"disc-assign-kernel-name", "ModuleOp"> {
  let summary = "Assign a meaningful name for each gpu kernel.";
  let constructor = "createDiscAssignKernelNamePass()";
}

def DiscLowerGpuOpsToNVVMOpsPass : Pass<"disc-convert-gpu-to-nvvm", "gpu::GPUModuleOp"> {
  let summary = "Generate NVVM operations for gpu operations";
  let constructor = "createDiscLowerGpuOpsToNVVMOpsPass()";
  let dependentDialects = ["NVVM::NVVMDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">,
    Option<"hasRedux", "has-redux", "bool",
           /*default=*/"false", "whether has redux.">,
  ];
}

def DiscLowerGpuOpsToROCDLOpsPass : Pass<"disc-convert-gpu-to-rocdl", "gpu::GPUModuleOp"> {
  let summary = "Generate ROCDL operations for gpu operations";
  let constructor = "createDiscLowerGpuOpsToROCDLOpsPass()";
  let dependentDialects = ["ROCDL::ROCDLDialect", "memref::MemRefDialect"];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned",
           /*default=kDeriveIndexBitwidthFromDataLayout*/"0",
           "Bitwidth of the index type, 0 to use size of machine word">
  ];
}

def UnhandledAtomicRMWConverterPass : Pass<"disc-convert-unhandled-atomic-rmw", "mlir::func::FuncOp"> {
  let summary = "Convert unhandled AtomicRMWOp to memref::GenericAtomicRMWOp.";
  let constructor = "createDiscUnhandledAtomicRMWConverterPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

def DiscShapeSimplifierPass : Pass<"disc-shape-simplifier", "ModuleOp"> {
  let summary = "propagate some known shape information.";
  let constructor = "createDiscShapeSimplifierPass()";
  let dependentDialects = ["disc_shape::DISCShapeDialect"];
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
    Option<"insert_tie_shape_", "insert-tie-shape", "bool",
           /*default=*/"false", "whether tie shape op is inserted.">,
  ];
}

def DiscMathApproximationPass : Pass<"disc-math-approximation", "mlir::func::FuncOp"> {
  let summary = "using approximation for some special math ops.";
  let constructor = "createDiscMathApproximationPass()";
  let dependentDialects = [
    "mlir::math::MathDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

def DiscFlattenMemrefAccessPass : Pass<"disc-flatten-memref-access", "mlir::func::FuncOp"> {
  let summary = "flatten memref access to its 1D format.";
  let constructor = "createDiscFlattenMemrefAccessPass()";
  let dependentDialects = [
    "mlir::disc_shape::DISCShapeDialect"
  ];
}

def Canonicalizer : Pass<"disc-canonicalize"> {
  let summary = "Canonicalize operations";
  let description = [{
    This pass performs various types of canonicalizations over a set of
    operations. See [Operation Canonicalization](Canonicalization.md) for more
    details.
  }];
  let constructor = "createDiscCanonicalizerPass()";
  let options = [
    // These two options provide filtering for which patterns are applied. These
    // should be passed directly to the FrozenRewritePatternSet when it is
    // created.
    ListOption<"disabledPatterns_", "disable-patterns", "std::string",
               "Labels of patterns that should be filtered out during"
               " application",
               "llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"enabledPatterns_", "enable-patterns", "std::string",
               "Labels of patterns that should be used during"
               " application, all other patterns are filtered out",
               "llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def DiscMemrefCanonicalizer : Pass<"disc-memref-canonicalize", "mlir::func::FuncOp"> {
  let summary = "Do some memref-related cleanup.";
  let constructor = "createDiscMemrefCanonicalizerPass()";
}

// todo: cpu host codegen
def DiscOutlineCpuKernel : Pass<"disc-outline-cpu-kernel", "ModuleOp"> {
  let summary = "outline each cpu kernel to a dedicated function.";
  let constructor = "createDiscOutlineCpuKernelPass()";
}

// todo: cpu host codegen
def DiscCpuMapParallelLoop : Pass<"disc-cpu-map-parallel-loop", "mlir::func::FuncOp"> {
  let summary = "assign a parallel schedule for each parallel op on cpu.";
  let constructor = "createDiscCpuMapParallelLoopPass()";
}

def DiscStitchFusion : Pass<"disc-stitch-fusion", "mlir::func::FuncOp"> {
  let summary = "Lowring stitch fusion pattern to parallel for loops.";
  let constructor = "createDiscStitchFusionPass()";
}

def TransposeSimplifierPass : Pass<"disc-transpose-simplifier", "mlir::func::FuncOp"> {
  let summary = "Remove some redundant transpose ops in the graph.";
  let constructor = "createTransposeSimplifierPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
  ];
}

def RalInjectExecutionContextPass : Pass<"disc-ral-inject-execution-context", "ModuleOp"> {
  let summary = "Inject DISC RAL execution context.";
  let constructor = "createRalInjectExecutionContextPass()";
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
  ];
}

def LhloFusionInlinerPass : Pass<"lhlo-fusion-inliner", "mlir::func::FuncOp"> {
  let summary = "Inline the contents of its body to the parent region "
    "after its body has been lowered";
  let constructor = "createLhloFusionInlinerPass()";
}

def DiscDotMergePass : Pass<"disc-dot-merge", "mlir::func::FuncOp"> {
  let summary = "Dot merge optimization. Either merge same-shape dots into "
                "batched dot, or merge dots with same operand into one (on-going)";
  let constructor = "createDiscDotMergePass()";
}

def DiscQuantizedDotMergePass : Pass<"disc-quantized-dot-merge", "mlir::func::FuncOp"> {
  let summary = "Dot merge optimization. Either merge same-shape dots into "
                "batched dot, or merge dots with same operand into one (on-going)";
  let constructor = "createDiscQuantizedDotMergePass()";
}

def DiscAlgebraicSimplifierPass : Pass<"disc-algebraic-simplifier", "mlir::func::FuncOp"> {
  let summary = "Apply some basic algebraic simplification optimizations";
  let constructor = "createDiscAlgebraicSimplifierPass()";
}

def DiscDynamicSliceConverterPass : Pass<"disc-dynamic-slice-converter", "mlir::func::FuncOp"> {
  let summary = "Convert mhlo.dynamic_slice with dynamic input to mhlo.real_dynamic_slice.";
  let constructor = "createDiscDynamicSliceConverterPass()";
  let dependentDialects = [
      "tensor::TensorDialect",
  ];
}

def DiscShapeOptimizationPass : Pass<"disc-shape-optimization", "ModuleOp"> {
  let summary = "apply some shape-related optimizations.";
  let constructor = "createDiscShapeOptimizationPass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
      "shape::ShapeDialect"
  ];
  let options = [
    Option<"entry_func_name_", "entry-func-name", "std::string",
           /*default=*/"\"main\"", "Name of the entry function.">,
    Option<"keep_tie_shape_", "keep-tie-shape", "bool",
           /*default=*/"false", "whether to keep tie shape op after this pass.">,
  ];
}

def DiscStripShapeConstraintOpsPass : Pass<"disc-strip-shape-constraint-ops", "ModuleOp"> {
  let summary = "strip disc shape constraint ops.";
  let constructor = "createStripShapeConstraintOpsPass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
  ];
}

def ForLoopUnrollInterleave : Pass<"disc-for-loop-unroll-interleave", "mlir::func::FuncOp"> {
  let summary = "Unroll SCF for loops and interleave the unrolling";
  let constructor = "createForLoopUnrollInterleavePass()";
}

def DiscDuplicateComputationForFusionPass : Pass<"disc-duplicate-computation-for-fusion", "mlir::func::FuncOp"> {
  let summary = "Duplicate some computations in order to enable more fusion opportunity and remove memory footprint.";
  let constructor = "createDiscDuplicateComputationForFusionPass()";
  let options = [
    Option<"gpu_enabled_", "gpu-enabled", "bool",
           /*default=*/"true", "whether gpu is available.">,
    Option<"fusion_strategy_", "fusion-strategy", "std::string",
           /*default=*/"\"base\"", "Name of the fusion strategy.">,
  ];
  let dependentDialects = [
      "memref::MemRefDialect",
  ];
}

def LLVMInsertValueSimplifierPass :
        Pass<"disc-llvm-insert-value-simplifier", "LLVM::LLVMFuncOp"> {
  let summary = "Eliminate unnecessary InsertValueOp and ExtractValueOp of LLVM dialect.";
  let constructor = "createLLVMInsertValueSimplifierPass()";
  let dependentDialects = [
      "LLVM::LLVMDialect",
  ];
}

def FunctionDeadArgumentEliminationPass :
        Pass<"disc-function-dead-argument-elimination", "gpu::GPUModuleOp"> {
  let summary = "Eliminate dead arguments of GPU LLVM function.";
  let constructor = "createFunctionDeadArgumentEliminationPass()";
  let dependentDialects = [
      "LLVM::LLVMDialect",
  ];
}

def SideEffectLoopInvariantCodeMotionPass :
        Pass<"disc-side-effect-loop-invariant-code-motion", "gpu::GPUFuncOp"> {
  let summary = "Apply loop invariant code motion on operators with side effect.";
  let constructor = "createSideEffectLoopInvariantCodeMotionPass()";
  let dependentDialects = [
      "gpu::GPUDialect",
  ];
}

def DiscDenseToSparsePass : Pass<"disc-dense-to-sparse", "mlir::func::FuncOp"> {
  let summary = "Convert dense gemm/conv to sparse gemm/conv";
  let constructor = "createDiscDenseToSparsePass()";
  let options = [
    Option<"enable_sparse_convert_", "enable-sparse-convert", "bool",
           /*default=*/"false", "Wether convert weight to sparse format.">,
  ];
}

def DiscSparseGemmTransposeSimplifierPass : Pass<"disc-sparse-gemm-transpose-simplifier", "mlir::func::FuncOp"> {
  let summary = "Remove some redundant transpose ops before sparse gemm/conv";
  let constructor = "createDiscSparseGemmTransposeSimplifierPass()";
}


def DiscBF16ExpansionPass : Pass<"disc-bf16-expansion", "mlir::func::FuncOp"> {
  let summary = "expand ext/trunc of bf16";
  let constructor = "createDiscBF16ExpansionPass()";
  let dependentDialects = [
    "arith::ArithDialect"
  ];
}

def DiscConvertFakeQuantOpPass : Pass<"disc-convert-fake-quant-op", "mlir::func::FuncOp"> {
  let summary = "Convert fake_quant annotated graph to the real quantized version.";
  let constructor = "createDiscConvertFakeQuantOpPass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
  ];
}

def DiscLowerQuantizeAndDequantizePass : Pass<"disc-lower-quantize-and-dequantize", "mlir::func::FuncOp"> {
  let summary = "Lower quantize and dequantize op to a bound of basic elementwise ops.";
  let constructor = "createDiscLowerQuantizeAndDequantizePass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
      "shape::ShapeDialect",
      "mhlo::MhloDialect",
  ];
}

def DiscTranformWeightDataLayoutForWeightOnlyQuantPass : Pass<"disc-transform-weight-data-layout-for-weight-only-quant", "mlir::func::FuncOp"> {
  let summary = "Transform weight data layout for faster-transformer's weight only gemm kernel";
  let constructor = "createDiscTranformWeightDataLayoutForWeightOnlyQuantPass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
      "shape::ShapeDialect",
      "mhlo::MhloDialect",
  ];
}

def DiscLowerGpuQuantizeAndDequantizePass : Pass<"disc-lower-gpu-quantize-and-dequantize", "mlir::func::FuncOp"> {
  let summary = "Lower quantize and dequantize op to a bound of basic elementwise ops on GPU.";
  let constructor = "createDiscLowerGpuQuantizeAndDequantizePass()";
  let dependentDialects = [
      "disc_shape::DISCShapeDialect",
      "shape::ShapeDialect",
      "mhlo::MhloDialect",
  ];
}

def DiscBufferDeallocationPass : Pass<"disc-buffer-deallocation", "mlir::func::FuncOp"> {
  let summary = "Insert dealloc op for some disc specific ops (e.g. custom_call_v2 op).";
  let constructor = "createDiscBufferDeallocationPass()";
}

def DiscCustomCallRewriterPass : Pass<"disc-custom-call-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite custom call ops according to its layout attribute.";
  let constructor = "createDiscCustomCallRewriterPass()";
  let dependentDialects = [
      "mhlo::MhloDialect",
  ];
}

def DiscCompIntensFusionToFuncPass : Pass<"disc-compute-intensive-fusion-to-func", "ModuleOp"> {
  let summary = "Pass to convert compute-intensive fusion to func-op.";
  let constructor = "createDiscCompIntensFusionToFuncPass()";
}

def DiscCompIntensFusionToCUDASourcePass : Pass<"disc-compute-intensive-fusion-to-cuda-source", "ModuleOp"> {
  let summary = "Pass to convert functions representing compute-intensive "
                "fusion into CUDA source code.";
  let constructor = "createDiscCompIntensFusionToCUDASourcePass()";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
  ];
}

def DiscGPUSourceToLibPass : Pass<"disc-gpu-source-to-lib", "ModuleOp"> {
  let summary = "Pass to convert CUDA code to library, possibly with host side logic.";
  let constructor = "createDiscGPUSourceToLibPass()";
  let options = [
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
  ];
}

def DiscTransformLegalizeToLoopPass : Pass<"disc-transform-legalize-to-loop", "mlir::func::FuncOp"> {
  let summary = "Legalize transfrom fusion pattern to loop.";
  let constructor = "createDiscTransformLegalizeToLoopPass()";
  let options = [
    Option<"gpuEnabled_", "gpu-enabled", "bool",
            /*default=*/"false", "whether gpu is available.">,
    Option<"transformFileName_", "transform-file-name", "std::string",
           /*default=*/"\"\"", "Filename of the transform schedule.">,
    Option<"cc_major_", "gpu-sm-cc-major", "int",
            /*default=*/"8", "gpu sm cc_major.">,
    Option<"cc_minor_", "gpu-sm-cc-minor", "int",
            /*default=*/"0", "gpu sm cc_minor.">,
    Option<"enableExpensiveChecks_", "enable-expensive-checks", "bool",
           /*default=*/"false", "perform expensive checks to better report errors in the transform IR.">,
  ];
}

def DiscSparseOpRewriterPass : Pass<"disc-sparse-op-rewriter", "mlir::func::FuncOp"> {
  let summary = "Rewrite sparse op for possible fusion";
  let constructor = "createDiscSparseOpRewriterPass()";
}

def DiscDuplicateComputationAfterFusionPass : Pass<"disc-duplicate-computation-after-fusion", "mlir::func::FuncOp"> {
  let summary = "Duplicate and fuse some computations into the consumer fusion op in order to remove memory footprint.";
  let constructor = "createDiscDuplicateComputationAfterFusionPass()";
  let dependentDialects = [
      "memref::MemRefDialect",
  ];
}

def DiscEraseBufferDeallocationPass : Pass<"disc-erase-buffer-deallocation", "mlir::gpu::GPUFuncOp"> {
  let summary = "Erase dealloc op for GPU func ops.";
  let constructor = "createDiscEraseBufferDeallocationPass()";
}

def DiscInputOutputAliasPass : Pass<"disc-input-output-alias", "ModuleOp"> {
  let summary = "Input and output alias information for buffer reuse";
  let constructor = "createDiscInputOutputAliasPass()";
}

def DiscReduceBufferLiveRangePass : Pass<"disc-reduce-buffer-live-range", "mlir::func::FuncOp"> {
  let summary = "reduce buffer live range";
  let constructor = "createDiscReduceBufferLiveRangePass()";
}

def DiscShapePropagatePass : Pass<"disc-shape-propagate", "ModuleOp"> {
  let summary = "shape analysis pass";
  let constructor = "createDiscShapePropagatePass()";
}