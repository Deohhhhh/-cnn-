import torch.nn as nn  # 导入PyTorch的神经网络模块

class simplecnn(nn.Module):  # 定义CNN类（规范建议：类名应使用驼峰式SimpleCNN）
    def __init__(self, num_class):  # 初始化方法（规范建议：参数命名用复数形式num_classes更合适）
        super().__init__()  # 调用父类nn.Module的初始化方法（规范：必须的继承操作）
        
        # 特征提取部分
        self.feature = nn.Sequential(  # 用Sequential容器组织特征提取层
            # 输入形状：(batch_size, 3, 224, 224)（假设原始输入为224x224）
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 第一个卷积层
            # 参数说明：
            # in_channels=3（RGB图像）
            # out_channels=16（卷积核数量）
            # kernel_size=3（3x3卷积核）
            # padding=1保持特征图尺寸不变（输出尺寸：16x224x224）
            
            nn.ReLU(),  # 激活函数（规范：注释建议在逗号后加空格）
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化（输出尺寸：16x112x112）
            
            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出尺寸：32x112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸：32x56x56
        )

        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Linear(32*56*56, 128),  # 全连接层（规范：运算符两侧加空格）
            # 输入尺寸需要匹配前面输出的展平尺寸（32*56*56=100352）
            nn.ReLU(),
            nn.Linear(128, num_class)  # 最终分类层（规范：建议使用num_classes命名）
        )

    def forward(self, x):  # 前向传播方法
        x = self.feature(x)  # 特征提取（输出形状：batch_size×32×56×56）
        x = x.view(x.size(0), -1)  # 展平操作（规范：视图操作建议用reshape替代view）
        # x.size(0)获取batch_size，-1自动计算维度（结果形状：batch_size×100352）
        x = self.classifier(x)  # 分类预测
        return x  # 返回logits（未经过softmax的概率）