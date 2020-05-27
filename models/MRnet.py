import torch
import torch.nn as nn
from torchvision import models

class MRnet(nn.Module):
    """MRnet uses pretrained resnet50 as a backbone to extract features
    """
    
    def __init__(self): # add conf file

        super(MRnet,self).__init__()

        # init three backbones for three axis
        self.axial = self._generate_resnet()
        self.coronal = self._generate_resnet()
        self.saggital = self._generate_resnet()

        self.fc = nn.Sequential(
            nn.Linear(in_features=3*2048,out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=2),
        )

    def forward(self,x): # TODO : see what to do ??
        """ Input is given in the form of `[image1, image2, image3]` where
        `image1 = [1, slices, 3, 224, 224]`. Note that `1` is due to the 
        dataloader assigning it a single batch. 
        """

        # squeeze the first dimension as there
        # is only one patient in each batch
        images = [torch.squeeze(img, dim=0) for img in x]

        image1 = self.axial(images[0]).view(-1,2048)
        image2 = self.coronal(images[1]).view(-1,2048)
        image3 = self.saggital(images[2]).view(-1,2048)

        image1 = torch.max(image1,dim=0,keepdim=True)[0]
        image2 = torch.max(image2,dim=0,keepdim=True)[0]
        image3 = torch.max(image3,dim=0,keepdim=True)[0]

        output = torch.cat([image1,image2,image3], dim=1)

        output = self.fc(output)

        # no need to take softmax here
        # as cross_entropy loss combines both softmax and NLL loss
        return output
    
    def _generate_resnet(self):
        """make all resnet params non-trainable, called automatically
        in `__init__` and then generate a Resnet50 model to be used as a backbone
        """
        # init resnet
        backbone = models.resnet50(pretrained=True)
        resnet_modules = list(backbone.children())

        # remove last layer of resnet
        body = nn.Sequential(*resnet_modules[:-1])
        
        # make params non trainable
        for x in body.parameters():
            x.requires_grad = False

        return body

    def _load_wieghts(self):
        """load pretrained weights"""
        pass

    def _save_model(self):
        """Dump the model weights to `cfg['weights']` dir"""
        pass
        
    
