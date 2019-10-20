import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt


#建立数据集，是一个u型离散点------------------------------------------------------------------
#下方unsqueeze是把后边linspace一维变成二维
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
y = x.pow(2) + 0.3*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

x,y=Variable(x),Variable(y)
# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

#建立神经网络------------------------------------------------------------------
class Net(torch.nn.Module):#继承
    def __init__(self,n_feature,n_hidden,n_output):
        super(Net,self).__init__()
        #接下来本案例特殊开始
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,1)#括号里输入、输出

    def forward(self,x):
        x = F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(n_feature=1,n_hidden=12,n_output=1)#意思输入1个，隐藏层10个，输出1个
# print(net)


#可视化训练过程------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.ion()   # 画图
plt.show()

#训练网络------------------------------------------------------------------
# optimizer 是训练的工具
# optim包含很多工具，下行是其中一个，lr是学习速率【learning rate】
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率
optimizer = torch.optim.Adam(net.parameters(), lr=0.2, betas=(0.9, 0.99)) #替换一个高效的试试
# 预测值和真实值的误差计算公式 (MSE均方差)
loss_func = torch.nn.MSELoss()      


for t in range(300):#训练200步
    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差，prediction是预测值，y是真实值
#下方优化三步
    optimizer.zero_grad()   # 清空上一步的残余更新参数值
    loss.backward()         # 误差反向传播, 计算参数更新值
    optimizer.step()        # 将参数更新值施加到 net 的 parameters 上
    if t % 3 == 0:#每3步记录一下
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
