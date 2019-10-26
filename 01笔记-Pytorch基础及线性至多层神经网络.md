# Pytorch背景知识
> [官网：](pytorch.org)
使用方：facebook  twitter nvidia
使用方式：

优点：GPU运算、tensor可代替numpy、构建自动求导神经网络、python优先、命令式体验、快速轻巧

tensor：基本单元，一种张量

> [常用的形式（官网内的详细划分）](https://pytorch.org/docs/stable/tensors.html)
- torch.floattensor
- torch.longtensor
 
> 学习内容点：
- pytorch的tensor 于numpy 的tensor相互转换
    - 需要注意 GPU 上的 Tensor 不能直接转换为 NumPy ndarray，需要使用.cpu()先将 GPU 上的 Tensor 转到 CPU 上
    - 注意左边变量名与右边操作名的区别
    - 可以放在不同的GPU的方法
    - 从GPU换回CPU的方法，尤其用numpy时要转到CPU
- 为啥要从torch转到numpy呢？因为torch数学运算方式少于np;numpy计算能力更牛逼一些；
- tensor 操作分为两种，**一种是数学运算，一种是高级操作**；
    - x = torch.ones(2, 2)
    - x = torch.ones(4, 4).float()
    - x[1:3, 1:3] = 2 #换元素
    - z = torch.add(x, y) 或z=x+y
    - x = torch.randn(4, 3)
    - sum_x = torch.sum(x, dim=1) # 沿着行对 x 求和
    - x = x.squeeze(0) # 减少第一维
    - x = torch.randn(3, 4, 5)
    - x = x.permute(1, 0, 2)
    - x.size()
    - x.shape
    - x = x.view(-1, 5) # 变化纬度
    - max_value, max_idx = torch.max(x, dim=1) #沿着行
    - sum_x = torch.sum(x, dim=1) # 沿着行
    - print(x,x.type())

- **Variable**
    - from torch.autograd import Variable #变量引入
    - x = Variable(x_tensor, requires_grad=True) # 如何包起来
    - 求 x 和 y 的梯度
        - x = Variable(x_tensor, requires_grad=True)
        - z=(x+2)**2
        - z.backward()
        - print(x.grad)
- **静态图动态图**（可见网盘朱庇特笔记）
    - 静态图先定义计算图（模型）不断使用这个模型，每次使用不需要重新定义
        - debug难
        - 运行快
    - 动态图每次构建生成模型
        - debug方便
        - 运行慢
- [**求导公式复习**](https://baike.baidu.com/item/%E5%AF%BC%E6%95%B0%E8%A1%A8/10889755?fr=aladdin)
- [**矩阵乘法复习**](https://baike.baidu.com/item/%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95/5446029?fr=aladdin)
- [**希腊字母表**](https://baike.baidu.com/item/%E5%B8%8C%E8%85%8A%E5%AD%97%E6%AF%8D/4428067?fr=aladdin)
## 简单模型——线性、二维分类
> 监督学习

- 需要人为定义好——标记数据集，送给模型去学习

> 非监督学习
与上边相反

> 线性模型
y=wx+b
- 梯度下降
    - 梯度就是导数
    - 多个变量就是每个变量求导数
    - 梯度下降法就是找最小的梯度 
    - 误差函数 {1/n * sum((yi'-yi)^2)}
    - w = w - lr*(梯度) 
    - 学习率lr解读：不能一步走全了误差，每一步都小一点，lr就是这些步子多小的参数

> Logistic 回归模型
「y=Sigmoid( ωx + b )」
- 学习五小时能过吗
- 包好看超没超1000
- 所以⬆️主要应对分类问题
- 比线性回归模型多了一个Sigmoid函数
    - 第一步计算wx+b
    - 第二步加sigmoid
- Loss函数 
- pytorch中是`F.sigmoid`
- pytorch自带
    - [sigmoid函数](https://pytorch.org/docs/stable/nn.functional.html#sigmoid)
    - [functions](https://pytorch.org/docs/stable/nn.functional.html#loss-functions)
    - [optim优化器-SGD](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD)
    - [二分类BCEWithLogitsLoss](https://pytorch.org/docs/stable/nn.html#bcewithlogitsloss)
    - 使用（下边的事没有用BCEWLloss公式的，所以loss是自己def的）：
```
def logistic_regression(x):
    return F.sigmoid(torch.mm(x,w)+b)

def binary_loss(y_pred,y):
    logits = (y*y_pred.clamp(1e-12).log()+(1-y)*(1-y_pred).clamp(1e-12).log()).mean()
    return -logits

optimizer = torch.optim.SGD([w,b],lr=1.)

for e in range(1000):
    y_pred = logistic_regression(x_data)
    loss = binary_loss(y_pred,y_data) #计算loss
    #更新参数
    optimizer.zero_grad() #使用优化器梯度归零
    loss.backward()
    optimizer.step() #使用优化器来更新参数，即修正w、b
    ………………
```

## 多层神经网络模型
- 多个输入x 到 线性方程 加成 Sigmoid→ y-hat
- 多层就是多个回归模型
- hidden layer 就是 linear + sigmoid 
- [激活函数](https://pytorch.org/docs/stable/nn.functional.html#non-linear-activation-functions)：[Sigmoid](https://pytorch.org/docs/stable/nn.html#sigmoid)、[ReLU](https://pytorch.org/docs/stable/nn.html#relu)、[Softmax](https://pytorch.org/docs/stable/nn.html#softmax)、SELU……
    - 为什么需要呢？
    - 因为比如多层不加激活函数，等价于一层的，没意义了
> 简单方法：Sequential  和 module
- Sequential
```
seq_net = nn.Sequential(
    nn.Linear(2, 4), # PyTorch中的线性层，wx+b,表输入2维，输出4维
    nn.Tanh(),       # 表激活函数
    nn.Linear(4, 1)  # 表输入4维度，输出1维度
)
```

- Module

看似比Sequential更加复杂，方式更灵活，因为有forward这层

```
class module_net(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(module_net, self).__init__()
        self.layer1 = nn.Linear(num_input, num_hidden)
        
        self.layer2 = nn.Tanh()
        
        self.layer3 = nn.Linear(num_hidden, num_output)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
```

## 深层神经网络
> MNIST 手写字案例
### 深度神经网络方式
- 10分类，Sigmoid函数不能再保证概率和为1
- softmax(zi)=e^zi / sum (e^zj) 
- softmax 把 n个输出，变成n个概率，其中最大的就是我们要确认的分类
- Loss函数需要适用任何多分类——**交叉熵**
    - 衡量pq相似度：CrossEntropy(p,q)=-1/m sum(p(x)*logq(x))
    - 二分类loss CE= -1/n sum(yi*log*y_hat + (1-yi)log(1-y_hat))
    - [pytorch内置了交叉熵-点击查看](https://pytorch.org/docs/stable/nn.html#crossentropyloss)
```
# 使用内置函数下载 mnist 数据集
train_set = mnist.MNIST('./data', train=True, download=True)
test_set = mnist.MNIST('./data', train=False, download=True)

# 设置训练集、测试集
train_data= DataLoader(train_set, batch_size=64, shuffle=True)
test_data= DataLoader(test_set, batch_size=128, shuffle=False)

# 适用Sequential 定义4层神经网络
net = nn.Sequential(
    nn.Linear(784,400),
    nn.ReLU,
    nn.Linear(400,200),
    nn.ReLU,
    nn.Linear(200,100),
    nn.ReLU,
    nn.Linear(100,10),
)

# 定义 loss 函数
criterion = nn.CrossEntropyLoss() #criterion是自己取的
optimizer = torch.optim.SGD(net.parameters()  # 变量名自己取的
```

## 反向传播算法
- 简单线性模型梯度= ∂loss / ∂ω
- 多层神经网络 =？？计算量太大且很不容易操作
    - 第一步：从输入计算神经网络——每层计算梯度
    - 第二步：从右往左，反向计算梯度
    - 第三步：修正每个节点梯度
    - 第四步：循环上边过程（导数的链式法则）
- 其他优化算法：
    - torch.optim.SGD(param)      #随机梯度下降
    - torch.optim.SGD(param momentum=0.9)  #基于动量梯度下降
    - torch.optim.Adagrad(param)  #自适应梯度下降
    - torch.optim.RMSprop(param)  #RMSprop方法
    - torch.optim.Adadelta(param) #Adadelta方法
    - torch.optim.Adam(param)     #Adam方法
