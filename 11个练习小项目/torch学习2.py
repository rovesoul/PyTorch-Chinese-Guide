'''
https://www.youtube.com/watch?v=KBdb53NrVAc&list=PLXO45tsB95cJxT0mL0P3-G0rBcLSvVkKH&index=7
numpy和 torch很像
变量学习
什么是 Variable 
在 Torch 中的 Variable 就是一个存放会变化的值的地理位置. 里面的值会不停的变化. 
就像一个裝鸡蛋的篮子, 鸡蛋数会不停变动. 
那谁是里面的鸡蛋呢, 自然就是 Torch 的 Tensor 咯. 
如果用一个 Variable 进行计算, 那返回的也是一个同类型的 Variable.
'''
import torch 
from torch.autograd import Variable  # torch 中 Variable 模块
#------------------------------------------------------------------

tensor=torch.FloatTensor([[1,2],[3,4]])
variable=Variable(tensor,requires_grad=True)  #true 计算节点梯度

# print(tensor)
# print(variable)

'''
到目前为止, 我们看不出什么不同, 但是时刻记住,
 Variable 计算时, 它在背景幕布后面一步步默默地搭建着一个庞大的系统, 叫做计算图, computational graph. 
 这个图是用来干嘛的? 原来是将所有的计算步骤 (节点) 都连接起来, 最后进行误差反向传递的时候, 
 一次性将所有 variable 里面的修改幅度 (梯度) 都计算出来, 而 tensor 就没有这个能力啦.
'''

#加计算过程
t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

# print(t_out)
# print(v_out)

v_out.backward()
#隐藏计算过程
# v_out=1/4*sum(var*var)
# d(v_out)/d(var)=1/4*2*variable=variable/2
print(variable.grad) #反向传递更新值是多少
'''
tensor([[0.5000, 1.0000],
        [1.5000, 2.0000]])
'''
print('variable',variable) #  Variable 形式
print('variable.data',variable.data) # tensor 形式
print('variable.data.numpy()',variable.data.numpy())  #variable.data是tensor形式,# 整体numpy 形式
