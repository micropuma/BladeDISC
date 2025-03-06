import os
import torch
import torch_blade

# set TORCH_BLADE_DEBUG_LOG=on
os.environ["TORCH_BLADE_DEBUG_LOG"] = "on"
# do BladeDISC optimization
w = h = 10
dnn = torch.nn.Sequential(
      torch.nn.Linear(w, h),
      torch.nn.ReLU(),
      torch.nn.Linear(h, w),
      torch.nn.ReLU()).cuda().eval()
with torch.no_grad():
  # BladeDISC torch_blade optimize will return an optimized TorchScript
  opt_dnn_ts = torch_blade.optimize(
    dnn, allow_tracing=True, model_inputs=(torch.ones(w, h).cuda(),))

# print optimized code
print(opt_dnn_ts.code)

# list the debug files dumped
