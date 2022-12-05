
import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights
#from torchvision.models.inception import inception_v3

class Swin_T(nn.Module):
  def __init__(self):
    super(Swin_T, self).__init__()
    self.model1 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    self.model2 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    self.model3 = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
   # self.model2 = resnet152(weights= ResNet152_Weights.IMAGENET1K_V2)
    self.conv2d = nn.Conv2d(1, 3, 3, stride=1, padding=1)
    # for param in self.model.layer1.parameters():  if frozen backbone required
    #  param.requires_grad == True
    self.fc1 = nn.Linear(3000, 1)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(128, 1)
  #  self.dropout = nn.Dropout(p =0.2)

  def forward(self, x, y, tda):

     x = self.model1(x)
     y = self.conv2d(y)
     y = self.model2(y)
     tda = self.model3(tda)

     z = torch.cat((x, y, tda),1)
     z = z.view(z.size(0), -1)
     z = self.fc1(z)
    #  z = self.relu(z)
    #  z = self.dropout(z)
    #  z = self.fc2(z)
     return z
 