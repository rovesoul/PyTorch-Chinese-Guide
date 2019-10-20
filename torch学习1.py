'''
https://www.youtube.com/watch?v=KBdb53NrVAc&list=PLXO45tsB95cJxT0mL0P3-G0rBcLSvVkKH&index=7
numpy和 torch很像
'''
import torch 
import numpy as np
#------------------------------------------------------------------
# np_data= np.arange(6).reshape((2,3))
# # 转换
# torch_data=torch.from_numpy(np_data)
# # 转换2
# tensor2array = torch_data.numpy()

# print(
#     '\nnumpy',np_data,
#     '\ntorch',torch_data,
#     '\ntensor2array',tensor2array
#     )
# abs------------------------------------------------------------------
# data=[-1,-2,1,2]
# tensor= torch.FloatTensor(data)
# print(
#     '\nabs',
#     '\nnumpy:',np.abs(data),
#     '\ntorch:',torch.abs(tensor)
# )
# sin------------------------------------------------------------------
# data=[-1,-2,1,2]
# tensor= torch.FloatTensor(data)
# print(
#     '\nsin',
#     '\nnumpy:',np.sin(data),
#     '\ntorch:',torch.sin(tensor)
# ) 
# 平均------------------------------------------------------------------
# data=[-1,-2,1,2]
# tensor= torch.FloatTensor(data)
# print(
#     '\nsin',
#     '\nnumpy:',np.mean(data),
#     '\ntorch:',torch.mean(tensor)
# ) 
# 矩阵------------------------------------------------------------------
data=[[1,2],[3,4]]
tensor=torch.FloatTensor(data) #32-bit floating point

print(
    '\n相乘',
    '\nnumpy:',np.matmul(data,data),
    '\ntorch:',torch.mm(tensor,tensor)
) 

# 矩阵------------------------------------------------------------------
