import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import matplotlib
from matplotlib import pyplot as plt

is_cuda = False
device = 'cpu'
if torch.cuda.is_available():
    is_cuda = True
    count = torch.cuda.device_count()
    device = torch.device(count - 1)

# 获取并封装数据
transformation = transforms.Compose([transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
train_dataset = datasets.MNIST('data/', train=True,
        transform=transformation, download=True)
test_dataset = datasets.MNIST('data/', train=False,
        transform=transformation, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=32, shuffle=True)

sample_data = next(iter(train_loader))

# 可视化图片数据
def plot_img(image):
    image = image.numpy()[0]
    mean = 0.1307
    std = 0.3081
    image = ((mean * image) + std)
    plt.imshow(image,cmap='gray')
    plt.show()

# plot_img(sample_data[0][2])
print('sample_data[0] size:', sample_data[0].size())
print('sample_data[1] size:', sample_data[1].size())

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), kernel_size=2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if is_cuda:
    model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)

#data, target = next(iter(train_loader))
#output = model(data.to(device))
#print('output.size:', output.size())
#print('target.size:', target.size())

def fit(epoch, model, data_loader, phase='training', volatile=True):
    if phase == 'training':
        model.train()
    elif phase == 'validation':
        model.eval()
        volatile = True
    running_loss = 0.0
    running_correct = 0

    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.to(device), target.to(device)
        if phase == 'training':
            optimizer.zero_grad()

        output = model(data)
        loss = F.nll_loss(output, target)

        running_loss += F.nll_loss(output, target, reduction='sum').item()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()

    loss = running_loss / len(data_loader.dataset)
    accuracy = (100.0 * running_correct.item()) / len(data_loader.dataset)

    print('{}: {} loss is {:5.4f} and accuracy is {}/{} {:10.4f}'.format(
        epoch, phase, loss, running_correct.item(),
        len(data_loader.dataset), accuracy))
    return loss, accuracy

train_losses, train_accuracy = [], []
val_losses, val_accuracy = [], []
for epoch in range(20):
    epoch_loss, epoch_accuracy = fit(epoch, model, train_loader, phase='training')
    val_epoch_loss, val_epoch_accuracy = fit(epoch, model, test_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(train_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)

#matplotlib.interactive(True)
plt.plot(range(1, len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1, len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
plt.show()

#plt.plot(range(1, len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
#plt.plot(range(1, len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
#plt.legend()

