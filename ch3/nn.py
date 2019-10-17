import torch
from torch import nn

# 定义网络
class FirstNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, output_size)

    def __forward__(self, input):
        out = self.l1(input)
        out = nn.ReLU(out)
        out = nn.l2(out)
        return out

# 损失函数
loss = nn.MSELoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5)
output = loss(input, target)
output.backward()
print(input.grad)

