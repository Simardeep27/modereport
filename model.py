import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms



class CustomVisual(nn.Module):
    def __init__(self,model_name='densenet201',pretrained=False):
        super(CustomVisual, self).__init__()
        self.model_name=model_name
        self.pretrained=pretrained
        self.model,self.out_features,self.avg_func,self.bn,self.linear=self.__get_model()
        self.activation=nn.ReLU


    def __get_model(self):
        model=None
        out_features=None
        func=None
        if self.model_name=='densenet201':
            densenet=models.densenet201(pretrained=self.pretrained)
            modules=list(densenet.features)
            model=nn.Sequential(*modules)
            func=nn.AvgPool2d(kernel_size=7,stride=1,padding=0)
            out_features=densenet.classifier.in_features

        linear=nn.Linear(in_features=out_features,out_features=out_features)
        bn=nn.BatchNorm1d(num_features=out_features,momentum=0.1)
        return model,out_features,func,linear,bn

    def forward(self, images):
        visual_features=self.model(images)
        avg_features=self.avg_func(visual_features).squeeze()
        return visual_features,avg_features

'''class MLC(nn.Module):
    def __init__(self,
                 classes=156,sementic_features_dim=512,
                 fc_in_features=2048,
                 k=10):
        super(MLC, self).__init__()
        self.classifier=nn.Linear(in_features=fc_in_features,out_features=classes)
        self.embed=nn.Embedding(classes,sementic_features_dim)
        self.k=k
        self.softmax=nn.Softmax()
        self.__init_weight()

    def __init_weight(self):
        self.classifier.weight.data.uniform_(-0.1,0.1)
        self.classifier.bias.data.fill_(0)

    def forward(self,avg_features):
        print('avg',avg_features)
        tags=self.softmax(self.classifier(avg_features))
        semantic_features=self.embed(torch.topk(tags,self.k)[1])
        return tags,semantic_features'''

class MLC(nn.Module):
    def __init__(self,
                 classes=156, sementic_features_dim=512,
                 fc_in_features=2048,k=10):
        super(MLC, self).__init__()
        self.net=nn.Sequential(
            
            nn.Conv2d(fc_in_features,classes,kernel_size=1,bias=False),
            nn.BatchNorm2d(classes),
            nn.ReLU(inplace=True),

            nn.Conv2d(int(fc_in_features/10),int(classes/10),kernel_size=1,bias=False),
            nn.BatchNorm2d(int(classes/10)),
            nn.ReLU(inplace=True),
        )
        self.classifier=nn.Linear(fc_in_features,classes)
        print(self.net)
        self.embed=nn.Embedding(classes,sementic_features_dim)
        self.k=k
        self.softmax=nn.Softmax()
#         self.__init_weight()

#     def __init_weight(self):
#         self.classifier.weight.data.uniform_(-0.1, 0.1)
#         self.classifier.bias.data.fill_(0)

    def forward(self,avg_features):
        batch=avg_features.shape[0]
        print(avg_features.shape)
        print(batch)
        avg_features=torch.reshape(avg_features,(avg_features.shape[0],avg_features.shape[1],1,1))
        avg_vals=self.net(avg_features)
        avg_vals = avg_vals.view(batch, -1)
        tags=self.softmax(self.classifier(avg_vals))
        semantic_features=self.embed(torch.topk(tags,self.k)[1])
        return tags,semantic_features














if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
#
    extractor = CustomVisual(model_name='densenet201',pretrained=False)
    tags=MLC(fc_in_features=extractor.out_features)
    tags=CustomMLC(fc_in_features=extractor.out_features)


