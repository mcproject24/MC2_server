import torch
from torch import optim
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
# import matplotlib.pyplot as plt


class CustomNet(nn.Module):
  def __init__(self):
    super(CustomNet, self).__init__()
    
    self.conv_layer_1 = nn.Sequential(
        nn.Conv2d(1, 16, 3, 1, 1),
        nn.ReLU(),
        # nn.MaxPool2d(kernel_size = 2)
        nn.BatchNorm2d(16)
    )

    self.conv_layer_2 = nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size = 2)
    )

    self.conv_layer_3 = nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size = 2),
        nn.Dropout(0.25)
    )


    self.linear_layer_1 = nn.Sequential(
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Dropout(0.2)
    )
    self.linear_layer_2 = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.1)
    )
    self.linear_layer_3 = nn.Sequential(
          nn.Linear(64, 10),
          nn.ReLU()
    )

  def forward(self, x):
    x = self.conv_layer_1(x)
    x = self.conv_layer_2(x)
    x = self.conv_layer_3(x)
    x = x.view(x.size(0), -1)
    x = self.linear_layer_1(x)
    x = self.linear_layer_2(x)
    out = self.linear_layer_3(x) # out contains the predictions for a batch of 50 images at a time so itll be a 50x10 tensor
    return out



def initialize_env(seed, gpu = False):
  random_seed = seed # CSE535 project group number
  
  torch.manual_seed(random_seed)
  if gpu:
    if not torch.cuda.is_available():
      print("Sorry, no cuda enabled GPU found in the system! Defaulting to cpu.")
      device = torch.device('cpu')
    else:
      print("GPU found, switching env to GPU")
      device = torch.device('cuda')
      torch.backends.cudnn.enabled = False
  else:
    device = torch.device('cpu')
  return device



def prepare_data():
  
  train_data = datasets.MNIST(root = "./data", train = True, download = True, transform = ToTensor())
  test_data = datasets.MNIST(root = "./data", train = False, download = True, transform = ToTensor())

  train_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
  test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True)

  return train_loader, test_loader


def train():
  model.train()
  print("Training Started...")
  for epoch in range(num_epochs): 
    for batch_idx, (images, labels) in enumerate(train_loader):
      images, labels = images.to(device), labels.to(device)
      out = model(images)
      loss = loss_fn(out, labels)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if (batch_idx+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                .format(epoch + 1, num_epochs, batch_idx + 1, len(train_loader), loss.item()))

def evaluate():
  model.eval()

  samples = len(test_loader) * batch_size
  score = 0
  for _, (images, labels) in enumerate(test_loader):
    images, labels = images.to(device), labels.to(device)
    print(images.shape)
    out = model(images)
    out_labels = torch.argmax(out, dim = 1)
    score += (out_labels == labels).sum().item()
  return score/samples

def save():
  torch.save(model.state_dict(), "./mnist_model.pth")

if __name__ == "__main__":
  device = initialize_env(seed = 24, gpu = True)

  model = CustomNet()
  model.to(device)
  
  batch_size = 50
  
  train_loader, test_loader = prepare_data()

  num_epochs = 100
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.SGD(model.parameters(), lr = 0.01)

  train()
  ACC = evaluate()
  print("Accuracy of the model: ", ACC)
  save()
