# 01 Pytorch背景知识
> [点击查看官网](pytorch.org)
使用方：facebook、twitter、nvidia
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
