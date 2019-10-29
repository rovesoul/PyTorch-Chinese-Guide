# 11 Pytorch中 通过微调进行迁移学习

pytorch 一直为我们内置了前面我们讲过的那些著名网络的预训练模型，不需要我们自己去 ImageNet 上训练了，模型都在 [torchvision.models(点击看官方文档)](https://pytorch.org/docs/0.3.0/torchvision/models.html#torchvision-models)里面，比如我们想使用预训练的 50 层 resnet，就可以用 `torchvision.models.resnet50(pretrained=True)` 来得到

比如这么输入来得到模型:其他内容详细看
```python
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()
squeezenet = models.squeezenet1_0()
densenet = models.densenet161()
inception = models.inception_v3()
```
