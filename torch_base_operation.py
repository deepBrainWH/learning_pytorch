import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # view函数改变形状,flatten
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def num_flat_features(self, x):
        # 除去批处理的其他维度
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())
input = torch.randn(1, 1, 32, 32)
target = torch.randn(1, 10)
# out = net(input)
criterion = nn.MSELoss()
# loss = criterion(out, target)
# print(loss)
# print(loss.grad_fn)
# net.zero_grad()
# print("conv1.bias.grad before backward")
# print(net.conv1.bias.grad)
# loss.backward()
# print("conv1.bias.grad after backward")
# print(net.conv1.bias.grad)
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data*learning_rate)
# Using different optimizer
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()