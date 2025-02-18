import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import CrossEntropyLoss
from src.iou_metric import IoULoss, IoUMetric
from torch.nn import Softmax
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
from src.utils import close_figures
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage, ToTensor
import os

def train_model(model: Module, device: str, loader: DataLoader, optimizer: Optimizer):
  model.train()
  model.to(device)

  cel = True
  if cel:
    criterion = CrossEntropyLoss(reduction='mean')
  else:
    criterion = IoULoss(softmax=True)

  running_loss = 0.0
  running_count = 0

  for batch_index, (inputs, targets) in enumerate(loader):
    optimizer.zero_grad()
    inputs: torch.Tensor = inputs.to(device)
    targets: torch.Tensor = targets.to(device)
    outputs = model(inputs)

    if cel:
      targets = targets.squeeze(dim=1)
    
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    running_count += targets.size(0)
    running_loss += loss.item()
  
  print("Trained {} samples, Loss: {:.4f}".format(running_count, running_loss / batch_index + 1))

def prediction_accuracy(ground_truth_labels: torch.Tensor, predicted_labels: torch.Tensor):
  equal = ground_truth_labels == predicted_labels
  return equal.sum().item() / predicted_labels.numel()

def print_test_dataset_masks(model: Module, device: str, 
                             pet_test_targets: torch.Tensor, pet_test_labels: torch.Tensor, 
                             epoch: int, save_path: str, show_plot: bool):
  model.eval()
  model.to(device)

  outputs: torch.Tensor = model(pet_test_targets.to(device))
  pet_test_labels = pet_test_labels.to(device)
  predictions: torch.Tensor = Softmax(dim=1)(outputs)

  predictions_labels = predictions.argmax(dim=1)
  predictions_labels = predictions_labels.unsqueeze(dim=1)
  predictions_masks = predictions_labels.to(dtype=torch.float)

  iou = MulticlassJaccardIndex(3, average="micro", ignore_index=1) # 1=background
  iou_accuracy = iou(predictions_masks, pet_test_labels)

  pixel_metric = MulticlassAccuracy(3, average="micro")
  pixel_accuracy = pixel_metric(predictions_labels, pet_test_labels)

  custom_iou = IoUMetric(outputs, pet_test_labels, softmax=True)

  title = 'Epoch: {:02d}, Accuracy[Pixel: {:.4f}, IoU: {:.4f}, Custom IoU: {:.4f}]'.format(epoch, pixel_accuracy, iou_accuracy,custom_iou)
  print(title)

  close_figures()

  fig = plt.figure(figsize=(10, 10))
  fig.suptitle(title)

  fig.add_subplot(3, 1, 1)
  plt.imshow(ToPILImage()(make_grid(pet_test_targets, nrow=7)))
  plt.axis('off')
  plt.title('Targets')

  fig.add_subplot(3, 1, 2)
  plt.imshow(ToPILImage()(make_grid(pet_test_labels.float() / 2.0, nrow=7)))
  plt.axis('off')
  plt.title('Ground Truth Labels')

  fig.add_subplot(3, 1, 3)
  plt.imshow(ToPILImage()(make_grid(predictions_masks.float() / 2.0, nrow=7)))
  plt.axis('off')
  plt.title('Predicted Labels')

  if save_path:
    plt.savefig(os.path.join(save_path, 'epoch_{:02d}.png'.format(epoch)), format="png", bbox_inches="tight", pad_inches=0.4)
  
  if show_plot:
    plt.show()
  else:
    close_figures()


