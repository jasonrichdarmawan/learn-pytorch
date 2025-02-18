from torch import Tensor, cat
from torch.nn import Softmax, Module

def IoUMetric(prediction: Tensor, ground_truth: Tensor, softmax: bool):
  if softmax:
    # if input is logits
    prediction = Softmax(dim=1)(prediction)
  
  # one-hot encoded masks for all 3 output channels
  ground_truth = cat([(ground_truth == i) for i in range(3)], dim=1)

  intersection = ground_truth * prediction
  union = ground_truth + prediction - intersection

  # sum over all pixels in the batch, except for the batch dimension
  iou = (intersection.sum(dim=(1,2,3)) + 0.001) / (union.sum(dim=(1,2,3)) + 0.001)

  # compute the mean over the batch dimension
  return iou.mean()

class IoULoss(Module):
  def __init__(self, softmax: bool):
    super().__init__()
    self.softmax = softmax

  def forward(self, prediction: Tensor, ground_truth: Tensor):
    return -(IoUMetric(prediction, ground_truth, self.softmax).log())