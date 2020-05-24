import torch
import torch.nn as nn
from torchvision import models

class MRnet(nn.Module):
    """MRnet uses pretrained resnet50 as a backbone to extract features
    """
    
    def __init__(self): # add conf file

        super(MRnet,self).__init__()

        # init resnet
        backbone = models.resnet50(pretrained=True)
        resnet_modules = list(backbone.children())

        # get the last conv layer of resnet
        self.body = nn.Sequential(*resnet_modules[:-2])

        # make body non-trainable
        self._set_grads()

        self.fc = nn.Sequential(
            nn.Linear(in_features=2048*7*7,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x): # TODO : see what to do ??
        """ input should be of form `[1,slices,3,224,224]` ,
        make sure to add 3 dimesnion to image by adding same image across
        all three RGB.
        """

        # squeeze the first dimension as there
        # is only one patient in each batch
        x = torch.squeeze(x, dim=0) 

        x = self.body(x)
        x = x.view(-1,2048*7*7) # flatten x
        x = self.fc(x)

        return x
    
    def _set_grads(self):
        """make all resnet params non-trainable, called automatically
        in `__init__`
        """
        
        for x in self.body.parameters():
            x.requires_grad = False

    def _load_wieghts(self):
        """load pretrained weights"""
        pass

    def _save_model(self):
        """Dump the model weights to `cfg['weights']` dir"""
        pass
        
    
