import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from torch.nn import CrossEntropyLoss
from src.iou_metric import IoULoss, IoUMetric
from torch.nn import Softmax
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage
import os

def train_model(model: Module, device: str, loader: DataLoader, 
                optimizer: Optimizer):
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

def evaluate_model(model: Module, device: str, 
                   images: torch.Tensor, ground_truths: torch.Tensor, 
                   epoch: int, root: str, show_plot: bool):
  model.eval()

  model.to(device=device)
  images = images.to(device=device)
  ground_truths = ground_truths.to(device=device)

  logits: torch.Tensor = model(images)                 # shape [batch_size, num_classes, height, width]
  probabilities: torch.Tensor = Softmax(dim=1)(logits) # shape [batch_size, num_classes, height, width]
                                                       # range [0,1] and sum to 1

  predictions_labels = probabilities.argmax(dim=1)                 # shape [batch_size, height, width]
  predictions_labels = predictions_labels.unsqueeze(dim=1)         # shape [batch_size, 1, height, width]
  predictions_masks = predictions_labels.to(dtype=torch.float)

  iou = MulticlassJaccardIndex(3, average="micro", ignore_index=1).to(device=device) # 1=background
  iou_accuracy = iou(predictions_masks, ground_truths)

  pixel_metric = MulticlassAccuracy(3, average="micro").to(device=device)
  pixel_accuracy = pixel_metric(predictions_labels, ground_truths)

  custom_iou = IoUMetric(probabilities, ground_truths, is_predictions_logits=False)

  title = 'Epoch: {:02d}, Accuracy[Pixel: {:.4f}, IoU: {:.4f}, Custom IoU: {:.4f}]'.format(epoch, pixel_accuracy, iou_accuracy,custom_iou)
  print(title)

  fig = plt.figure(figsize=(10, 10))
  fig.suptitle(title)

  fig.add_subplot(3, 1, 1)
  plt.imshow(ToPILImage()(make_grid(images, nrow=7)))
  plt.axis('off')
  plt.title('Targets')

  fig.add_subplot(3, 1, 2)
  plt.imshow(ToPILImage()(make_grid(ground_truths.float() / 2.0, nrow=7)))
  plt.axis('off')
  plt.title('Ground Truth Labels')

  fig.add_subplot(3, 1, 3)
  plt.imshow(ToPILImage()(make_grid(predictions_masks.float() / 2.0, nrow=7)))
  plt.axis('off')
  plt.title('Predicted Labels')

  if root:
    os.makedirs(os.path.join(root, 'epochs'), exist_ok=True)
    plt.savefig(os.path.join(root, 'epochs/epoch_{:02d}.png'.format(epoch)), format="png", bbox_inches="tight", pad_inches=0.4)
  
  if show_plot == False:
    plt.close(fig)

def train_loop(model: Module, device: str, trainval_loader: DataLoader, 
               test_data: tuple[torch.Tensor, torch.Tensor], epochs: tuple[int, int],
               optimizer: Optimizer, scheduler: lr_scheduler.LRScheduler, 
               root: str):
  test_images, test_ground_truths = test_data
  epoch_i, epoch_j = epochs
  for i in range(epoch_i, epoch_j):
    epoch = i
    print("Epoch: {:02d}, Learning Rate: {}".format(epoch, 
                                                    optimizer.param_groups[0]['lr']))
    train_model(model, device, trainval_loader, optimizer)
    with torch.inference_mode():
      evaluate_model(model, device, test_images, test_ground_truths, 
                     epoch=epoch, root=root, 
                     show_plot=(epoch == epoch_j - 1))

    if scheduler:
      scheduler.step()
    print("")

def validate_model(model: Module, device: str, loader: DataLoader):
  model.eval()
  model.to(device)

  iou = MulticlassJaccardIndex(3, average="micro", ignore_index=1).to(device=device)
  pixel_metric = MulticlassAccuracy(3, average="micro").to(device=device)

  iou_accuracies: list[float] = []
  pixel_accuracies: list[float] = []
  custom_iou_accuracies: list[float] = []

  print("Model parameters: {:.2f}M".format((sum(param.numel() for param in model.parameters())) / 1e6))

  for batch_index, (images, ground_truths) in enumerate(loader):
    images = images.to(device)
    ground_truths = ground_truths.to(device)

    logits: torch.Tensor = model(images)
    probabilities: torch.Tensor = Softmax(dim=1)(logits)
    predictions_labels = probabilities.argmax(dim=1)
    predictions_labels = predictions_labels.unsqueeze(dim=1)
    predictions_masks = predictions_labels.to(dtype=torch.float)

    iou_accuracy: torch.Tensor = iou(predictions_masks, ground_truths)
    pixel_accuracy: torch.Tensor = pixel_metric(predictions_labels, ground_truths)
    custom_iou: torch.Tensor = IoUMetric(probabilities, ground_truths, is_predictions_logits=False)

    iou_accuracies.append(iou_accuracy.item())
    pixel_accuracies.append(pixel_accuracy.item())
    custom_iou_accuracies.append(custom_iou.item())

    del images
    del ground_truths
    del logits
  
  iou_tensor = torch.FloatTensor(iou_accuracies)
  pixel_tensor = torch.FloatTensor(pixel_accuracies)
  custom_iou_tensor = torch.FloatTensor(custom_iou_accuracies)

  print("Pixel Accuracy: {:.4f}, IoU Accuracy: {:.4f}, Custom IoU Accuracy: {:.4f}".format(pixel_tensor.mean().item(), iou_tensor.mean().item(), custom_iou_tensor.mean().item()))

  

