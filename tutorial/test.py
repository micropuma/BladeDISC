import torch
import torch_blade
class MyCell(torch.nn.Module):
    def __init__(self):
        super(MyCell, self).__init__()
        self.c = torch.randn(10, 3)

    def forward(self, x, y):
        t1 = x + y
        t2 = torch.matmul(t1, self.c)
        t3 = torch.sum(t2)
        return t3 * t3

my_cell = MyCell()
x = torch.rand(10, 10)
y = torch.rand(10, 10)

with torch.no_grad():
    blade_cell = torch_blade.optimize(my_cell, allow_tracing=True, model_inputs=(x, y))

print(blade_cell(x, y))
