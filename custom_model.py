import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

class CustomMLC(nn.Module):
    def __init__(self,classes=156,sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(CustomMLC, self).__init__()
        self.classifier=nn.Linear(in_features=fc_in_features,out_features=classes)
        self.embed=nn.Embedding(classes,sementic_features_dim)
        self.k=k
        self.softmax=nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1,0.1)
        self.classifier.bias.data.fill_(0)

    def Forward(self,avg_features):

