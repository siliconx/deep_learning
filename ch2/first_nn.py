import torch
import numpy as np

def get_data():
    X = torch.Tensor([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
        7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]).view(17, 1)
    y = torch.Tensor([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
        2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
    return X, y

def get_weights():
    w = torch.randn(1, requires_grad=True)
    b = torch.randn(1, requires_grad=True)
    return w, b

def simple_network(x):
    y_pred = torch.matmul(x, w) + b
    return y_pred

def loss_fn(y, y_pred):
    loss = (y_pred - y).pow(2).sum()
    for param in [w, b]:
        if param.grad:  # 最开始为None
            param.grad.zero_()
        loss.backward()
        return loss.item()

def optimize(learning_rate):
    w.data -= learning_rate * w.grad
    b.data -= learning_rate * b.grad

learning_rate = 1e-4
x, y = get_data()  # x - 训练数据，y - 目标变量
w, b = get_weights()  # w, b - 学习参数

for i in range(1000):
    y_pred = simple_network(x)  # 计算wx + b的函数
    loss = loss_fn(y, y_pred)  # 计算y和y_pred的平方差的和
    optimize(learning_rate)  # 调整w, b，将损失最小化
    if i % 20 == 0:
        print('No.%05d' % i, loss)
