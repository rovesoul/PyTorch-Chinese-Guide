"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 1.3.1
不太明白的是，3.6.5报错，3.7.4通过
"""
import torch
import torch.utils.data as Data

torch.manual_seed(1)    # reproducible 这是为了重复试验的需要, 固定下伪随机数

BATCH_SIZE = 5  #一小批几个的意思
# BATCH_SIZE = 8

x = torch.linspace(1, 10, 10)       # 生成数据this is x data (torch tensor)
y = torch.linspace(10, 1, 10)       # 生成数据this is y data (torch tensor)
# print(x,y)
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)  #放入数据库,data意思训练数据,target算误差的
torch_dataset = Data.TensorDataset(x,y)  #放入数据库,data意思训练数据,target算误差的
# 变成一小批一小批的 
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size一小批几个的意思
    shuffle=True,               # random shuffle for training 要不要训练时候随机打乱
    num_workers=2,              # subprocesses for loading data 每次loader时用几线程
)


def show_batch():
    for epoch in range(3):   # train entire dataset 3 times
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            # train your data...
            print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
                  batch_x.numpy(), '| batch y: ', batch_y.numpy())


if __name__ == '__main__':
    show_batch()

