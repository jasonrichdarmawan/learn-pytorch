from torch import Tensor, cat
from torch.nn import Softmax, Module

def IoUMetric(predictions: Tensor, ground_truths: Tensor, is_predictions_logits: bool):
  """
  predictions  : shape [batch_size, 3, height, width]
  ground_truths: shape [batch_size, 1, height, width]
                 ground_truths \in {0, 1, 2}
  """
  if is_predictions_logits:
    # if output is logits
    predictions = Softmax(dim=1)(predictions)
  
  # one-hot encoded masks for all 3 output channels
  ground_truths = cat([(ground_truths == i) for i in range(3)], dim=1)

  intersection = ground_truths * predictions
  union = ground_truths + predictions - intersection

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