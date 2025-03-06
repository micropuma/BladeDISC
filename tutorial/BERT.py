import torch
import torch_blade
import time

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)

############################################# download model from huggingface #############################################
model_path = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda().eval()

def plain_tokenizer(inputs_str, return_tensors):
    inputs = tokenizer(inputs_str, return_tensors=return_tensors, padding=True)
    inputs = {key: value.cuda() for key, value in inputs.items()}
    
    # torch_blade.optimize 不支持 None 作为输入
    if "token_type_ids" in inputs and inputs["token_type_ids"] is None:
        del inputs["token_type_ids"]

    return (inputs['input_ids'], inputs['attention_mask'], inputs.get('token_type_ids', None))

class PlainTextClassificationPipeline(TextClassificationPipeline):
    def _forward(self, model_inputs):
        return self.model(*model_inputs)

classifier = pipeline(
    'sentiment-analysis',
    model=model,
    tokenizer=plain_tokenizer,
    pipeline_class=PlainTextClassificationPipeline,
    device=0
)

input_strs = [
    "We are very happy to show you the story.",
    "We hope you don't hate it."
]

results = classifier(input_strs)

for inp_str, result in zip(input_strs, results):
    print(inp_str)
    print(f" label: {result['label']}, with a score: {round(result['score'], 4)}")

############################################# Use BladeDISC for optimization #############################################
inputs_str = "Hey, the cat is cute."
inputs = plain_tokenizer(inputs_str, return_tensors="pt")

torch_config = torch_blade.config.Config()
torch_config.enable_mlir_amp = False  # disable mix-precision

# Ensure inputs are properly formatted for optimization
model_inputs = tuple(i for i in inputs if i is not None)

with torch.no_grad(), torch_config:
    optimized_ts = torch_blade.optimize(model, allow_tracing=True, model_inputs=model_inputs)

# Move optimized model to CUDA
optimized_ts = optimized_ts.cuda()

# Save the optimized TorchScript model
torch.jit.save(optimized_ts, "opt.disc.pt")

############################################# testbench #############################################
@torch.no_grad()
def benchmark(model, inputs, num_iters=1000):
    for _ in range(10):
        model(*inputs)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(num_iters):
        model(*inputs)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) / num_iters * 1000.0

def bench_and_report(input_strs):
    inputs = plain_tokenizer(input_strs, return_tensors="pt")
    model_inputs = tuple(i for i in inputs if i is not None)

    avg_latency_baseline = benchmark(model, model_inputs)
    avg_latency_bladedisc = benchmark(optimized_ts, model_inputs)

    print(f"Seqlen: {[len(s) for s in input_strs]}")
    print(f"Baseline: {avg_latency_baseline:.4f} ms")
    print(f"BladeDISC: {avg_latency_bladedisc:.4f} ms")
    print(f"BladeDISC speedup: {avg_latency_baseline / avg_latency_bladedisc:.4f}")

input_strs = [
    "We are very happy to show you the story.",
    "We hope you don't hate it."
]

bench_and_report(input_strs)

