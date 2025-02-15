# PINN-Spring-Mass-Damper-System
PINN弹簧质量抑制系统

原文：https://medium.com/@oladayo_7133/applying-deep-learning-in-physics-solving-a-spring-mass-damper-system-using-physics-informed-1a002474235a

![image](https://github.com/srrdhy/PINN-Spring-Mass-Damper-System/blob/main/images/%E6%96%B9%E7%A8%8B%E6%8F%8F%E8%BF%B0.png)
原文用了TensorFlow，我用Pytorch重写后效果不佳，损失无法下降，后发现网络没做初始化，初始化对训练的影响还是很大的。

关键的初始化代码：（在网络定义里）
```python
# 添加权重初始化
        for name, param in self.layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
```
相同参数配置下，无初始化的训练结果：
![[无初始化结果.png](https://github.com/srrdhy/PINN-Spring-Mass-Damper-System/blob/main/images/%E6%97%A0%E5%88%9D%E5%A7%8B%E5%8C%96%E7%BB%93%E6%9E%9C.png?raw=true)]
初始化后，与原文TensorFlow版本的结果类似，与解析解基本一致：
![[初始化后结果.png](https://github.com/srrdhy/PINN-Spring-Mass-Damper-System/blob/main/images/%E5%88%9D%E5%A7%8B%E5%8C%96%E5%90%8E%E7%BB%93%E6%9E%9C.png?raw=true)]
解析解：
![[解析解.png](https://github.com/srrdhy/PINN-Spring-Mass-Damper-System/blob/main/images/%E8%A7%A3%E6%9E%90%E8%A7%A3.png?raw=true)]
