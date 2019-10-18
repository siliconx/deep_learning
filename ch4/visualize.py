import torch
import numpy as np
import matplotlib.pyplot as plt

def show_tensor(t):
    plt.imshow(t.numpy())
    plt.show()

if __name__ == '__main__':
    t = torch.rand(64, 64) * 256
    show_tensor(t)

