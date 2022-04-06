import torch
import torch.nn as nn

class ExampleNet(nn.Module):

    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv2d_3 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 1)
        self.reLU_4 = nn.ReLU(inplace = False)
        self.maxPool2D_5 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2d_6 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
        self.reLU_11 = nn.ReLU(inplace = False)
        self.maxPool2D_12 = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, return_indices = False, ceil_mode = False)
        self.conv2d_7 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 0, dilation = 1, groups = 1, bias = True)
        self.reLU_9 = nn.ReLU(inplace = False)
        self.linear_14 = nn.Linear(in_features = 3*3*128, out_features = 128, bias = True)
        self.reLU_10 = nn.ReLU(inplace = False)
        self.linear_15 = nn.Linear(in_features = 128, out_features = 10, bias = True)

    def forward(self, x_para_1):
        x_conv2d_3 = self.conv2d_3(x_para_1)
        x_reLU_4 = self.reLU_4(x_conv2d_3)
        x_maxPool2D_5 = self.maxPool2D_5(x_reLU_4)
        x_conv2d_6 = self.conv2d_6(x_maxPool2D_5)
        x_reLU_11 = self.reLU_11(x_conv2d_6)
        x_maxPool2D_12 = self.maxPool2D_12(x_reLU_11)
        x_conv2d_7 = self.conv2d_7(x_maxPool2D_12)
        x_reLU_9 = self.reLU_9(x_conv2d_7)
        x_reshape_16 = torch.reshape(x_reLU_9,shape = (-1,3*3*128))
        x_linear_14 = self.linear_14(x_reshape_16)
        x_reLU_10 = self.reLU_10(x_linear_14)
        x_linear_15 = self.linear_15(x_reLU_10)
        return x_linear_15

class CNN(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN, self).__init__()
        # 建立第一个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv1 = nn.Sequential(
            # 第一个卷积con2d
            nn.Conv2d(  # 输入图像大小(1,28,28)
                in_channels=1,  # 输入图片的高度，因为minist数据集是灰度图像只有一个通道
                out_channels=16,  # n_filters 卷积核的高度
                kernel_size=3,  # filter size 卷积核的大小 也就是长x宽=5x5
                stride=1,  # 步长
                padding=1,  # 想要con2d输出的图片长宽不变，就进行补零操作 padding = (kernel_size-1)/2
            ),  # 输出图像大小(16,28,28)
            # 激活函数
            nn.ReLU(),
            # 池化，下采样
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样   ,stride 默认kernel_size
            # 输出图像大小(16,14,14)
        )
        # 建立第二个卷积(Conv2d)-> 激励函数(ReLU)->池化(MaxPooling)
        self.conv2 = nn.Sequential(
            # 输入图像大小(16,14,14)
            nn.Conv2d(  # 也可以直接简化写成nn.Conv2d(16,32,5,1,2)
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            # 输出图像大小 (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 输出图像大小(32,7,7)
        )
        # self.conv3 = nn.Sequential(   #使准确率变小
        #
        #     nn.MaxPool2d(2),
        #     # 输出图像大小(32,7,7)
        # )
        # 建立全卷积连接层
        self.out = nn.Linear(32 * 50 * 50, 200)  # 输出是10个类
        #self.out2 = nn.Linear(4096, 200)
    # 下面定义x的传播路线#
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        #x = self.conv3(x)  # 再通过conv2
        # 把每一个批次的每一个输入都拉成一个维度，即(batch_size,32*7*7)
        # 因为pytorch里特征的形式是[bs,channel,h,w]，所以x.size(0)就是batchsize
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        #x = self.out(x)
        output = self.out(x)
        return output
class CNN_500(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self):
        super(CNN_500, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 输出图像大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 1, 1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        # )
        self.out = nn.Linear(32 * 125 * 125, 100)  # 输出

        #self.out2 = nn.Linear(500, 5)  # 输出
        #self.out3 = nn.Linear(400, 29)  # 输出
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        #x = self.conv3(x)  # 再通过conv3
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        #x = self.out(x)
        output = self.out(x)
        #output = self.out3(x)
        return output
class CNN_500_part(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self,output_dim):
        super(CNN_500_part, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 输出图像大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 125 * 125, output_dim)  # 输出

        # self.out2 = nn.Linear(1000, 29)  # 输出
        #self.out3 = nn.Linear(400, 29)  # 输出
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        #x = self.out(x)
        output = self.out(x)
        #output = self.out3(x)
        return output
class CNN_1000_part_4(nn.Module):  # 我们建立的CNN继承nn.Module这个模块
    def __init__(self,output_dim):
        super(CNN_1000_part_4, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 输出图像大小
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 在2x2空间下采样
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 62 * 62, output_dim)  # 输出

        # self.out2 = nn.Linear(1000, 29)  # 输出
        #self.out3 = nn.Linear(400, 29)  # 输出
    def forward(self, x):
        x = self.conv1(x)  # x先通过conv1
        x = self.conv2(x)  # 再通过conv2
        x = self.conv3(x)  # 再通过conv3
        x = self.conv4(x)  # 再通过conv3
        x = x.view(x.size(0), -1)  # view就是把x弄成batchsize行个tensor
        #x = self.out(x)
        output = self.out(x)
        #output = self.out3(x)
        return output