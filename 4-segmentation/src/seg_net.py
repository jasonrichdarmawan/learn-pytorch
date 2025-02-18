from torch import Tensor, Size
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, MaxUnpool2d

class DownConv2(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      Conv2d(in_channels=in_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      Conv2d(in_channels=out_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.maxpool = MaxPool2d(kernel_size=2, return_indices=True)
  
  def forward(self, x: Tensor):
    y = self.sequential(x)
    pool_shape = y.shape
    y, indices = self.maxpool(y)
    return y, indices, pool_shape
  
class DownConv3(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      Conv2d(in_channels=in_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      Conv2d(in_channels=out_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
      Conv2d(in_channels=out_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU(),
    )
    self.max_pool = MaxPool2d(kernel_size=2, return_indices=True)
  
  def forward(self, x: Tensor):
    y = self.sequential(x)
    pool_shape = y.shape
    y, indices = self.max_pool(y)
    return y, indices, pool_shape

class UpConv2(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      Conv2d(in_channels=in_channels, out_channels=in_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      Conv2d(in_channels=in_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU()
    )
    self.max_unpool = MaxUnpool2d(kernel_size=2)
  
  def forward(self, x: Tensor, indices: Tensor, pool_shape: Size):
    y = self.max_unpool(x, indices=indices, output_size=pool_shape)
    y = self.sequential(y)
    return y
  
class UpConv3(Module):
  def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
    super().__init__()
    self.sequential = Sequential(
      Conv2d(in_channels=in_channels, out_channels=in_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      Conv2d(in_channels=in_channels, out_channels=in_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=in_channels),
      ReLU(),
      Conv2d(in_channels=in_channels, out_channels=out_channels, 
             kernel_size=kernel_size, padding=kernel_size//2, bias=False),
      BatchNorm2d(num_features=out_channels),
      ReLU()
    )
    self.max_unpool = MaxUnpool2d(kernel_size=2)
  
  def forward(self, x: Tensor, indices: Tensor, pool_shape: Size):
    y = self.max_unpool(x, indices=indices, output_size=pool_shape)
    y = self.sequential(y)
    return y
  
class ImageSegmentation(Module):
  def __init__(self, num_classes: int, kernel_size: int):
    super().__init__()
    self.num_classes = num_classes
    self.batch_norm = BatchNorm2d(num_features=3) # 3 channels for RGB images
    self.down_conv1 = DownConv2(in_channels=3, out_channels=64, kernel_size=kernel_size)
    self.down_conv2 = DownConv2(in_channels=64, out_channels=128, kernel_size=kernel_size)
    self.down_conv3 = DownConv3(in_channels=128, out_channels=256, kernel_size=kernel_size)
    self.down_conv4 = DownConv3(in_channels=256, out_channels=512, kernel_size=kernel_size)
    # self.down_conv5 = DownConv3(in_channels=512, out_channels=512, kernel_size=kernel_size)

    # self.up_conv5 = UpConv3(in_channels=512, out_channels=512, kernel_size=kernel_size)
    self.up_conv4 = UpConv3(in_channels=512, out_channels=256, kernel_size=kernel_size)
    self.up_conv3 = UpConv3(in_channels=256, out_channels=128, kernel_size=kernel_size)
    self.up_conv2 = UpConv2(in_channels=128, out_channels=64, kernel_size=kernel_size)
    self.up_conv1 = UpConv2(in_channels=64, out_channels=self.num_classes, kernel_size=kernel_size)
  
  def forward(self, batch: Tensor):
    """
    batch: Tensor of shape (batch_size, channels, height, width)
    output: Tensor of shape (batch_size, num_classes, height, width)
    """
    x = self.batch_norm(batch)

    # SegNet Encoder
    x, max_pool_indices1, pool_shape1 = self.down_conv1(x)
    x, max_pool_indices2, pool_shape2 = self.down_conv2(x)
    x, max_pool_indices3, pool_shape3 = self.down_conv3(x)
    x, max_pool_indices4, pool_shape4 = self.down_conv4(x)
    # Images are 128x128 in dimension.
    # After 4 max-pooling layers, the dimension is 128/16 = 8x8.
    # If we do another max-pooling layer, 
    # the dimension is 8/2 = 4z4. Consequently, we may lose too much spatial information
    # x, max_pool_indices5, pool_shape5 = self.down_conv5(x)

    # SegNet Decoder
    # x = self.up_conv5(x, max_pool_indices5, pool_shape5)
    x = self.up_conv4(x, max_pool_indices4, pool_shape4)
    x = self.up_conv3(x, max_pool_indices3, pool_shape3)
    x = self.up_conv2(x, max_pool_indices2, pool_shape2)
    x = self.up_conv1(x, max_pool_indices1, pool_shape1)

    return x