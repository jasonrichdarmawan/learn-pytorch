from torch.nn import Module, Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d
from torch import Tensor, Size

class DepthwiseSeparableConv2d(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
               padding: int, bias: bool):
    super().__init__()
    self.depthwise = Conv2d(in_channels=in_channels, out_channels=in_channels,
                            kernel_size=kernel_size, padding=padding, groups=in_channels, bias=bias)
    self.pointwise = Conv2d(in_channels=in_channels, out_channels=out_channels,
                            kernel_size=1, bias=bias)
  
  def forward(self, x: Tensor):
    x = self.depthwise(x)
    x = self.pointwise(x)
    return x

class DownDSConv2(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.max_pool = MaxPool2d(kernel_size=2, return_indices=True)
  
  def forward(self, x: Tensor):
    y = self.sequential(x)
    pool_shape = y.shape
    y, indices = self.max_pool(y)
    return y, indices, pool_shape

class DownDSConv3(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=out_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.max_pool = MaxPool2d(kernel_size=2, return_indices=True)
  
  def forward(self, x: Tensor):
    y = self.sequential(x)
    pool_shape = y.shape
    y, indices = self.max_pool(y)
    return y, indices, pool_shape

class UpDSConv2(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.max_unpool = MaxUnpool2d(kernel_size=2)
  
  def forward(self, x: Tensor, indices: Tensor, pool_shape: Size):
    y = self.max_unpool(x, indices, output_size=pool_shape)
    y = self.sequential(y)
    return y

class UpDSConv3(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=in_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      DepthwiseSeparableConv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=kernel_size//2,
                               bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.max_unpool = MaxUnpool2d(kernel_size=2)
  
  def forward(self, x: Tensor, indices: Tensor, pool_shape: Size):
    y = self.max_unpool(x, indices, output_size=pool_shape)
    y = self.sequential(y)
    return y

class ImageSegmentationDSC(Module):
  def __init__(self, num_classes: int, kernel_size: int):
    super().__init__()
    self.out_channels = num_classes
    self.batch_norm = BatchNorm2d(num_features=3) # 3 channels for RGB
    self.down_conv1 = DownDSConv2(in_channels=3, out_channels=64, kernel_size=kernel_size)
    self.down_conv2 = DownDSConv2(in_channels=64, out_channels=128, kernel_size=kernel_size)
    self.down_conv3 = DownDSConv3(in_channels=128, out_channels=256, kernel_size=kernel_size)
    self.down_conv4 = DownDSConv3(in_channels=256, out_channels=512, kernel_size=kernel_size)
    # self.down_conv5 = DownDSConv3(in_channels=512, out_channels=512, kernel_size=kernel_size)

    # self.up_conv5 = UpDSConv3(in_channels=512, out_channels=512, kernel_size=kernel_size)
    self.up_conv4 = UpDSConv3(in_channels=512, out_channels=256, kernel_size=kernel_size)
    self.up_conv3 = UpDSConv3(in_channels=256, out_channels=128, kernel_size=kernel_size)
    self.up_conv2 = UpDSConv2(in_channels=128, out_channels=64, kernel_size=kernel_size)
    self.up_conv1 = UpDSConv2(in_channels=64, out_channels=num_classes, kernel_size=kernel_size)

  def forward(self, x: Tensor):
    x = self.batch_norm(x)

    # SegNet Encoder
    x, max_pool_indices1, pool_shape1 = self.down_conv1(x)
    x, max_pool_indices2, pool_shape2 = self.down_conv2(x)
    x, max_pool_indices3, pool_shape3 = self.down_conv3(x)
    x, max_pool_indices4, pool_shape4 = self.down_conv4(x)
    # x, max_pool_indices5, pool_shape5 = self.down_conv5(x)

    # SegNet Decoder
    # x = self.up_conv5(x, max_pool_indices5, pool_shape5)
    x = self.up_conv4(x, max_pool_indices4, pool_shape4)
    x = self.up_conv3(x, max_pool_indices3, pool_shape3)
    x = self.up_conv2(x, max_pool_indices2, pool_shape2)
    x = self.up_conv1(x, max_pool_indices1, pool_shape1)

    return x