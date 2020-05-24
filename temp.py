# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import MRData

print("Loading Data...")
data_train = MRData(plane='axial',task='acl')

image,label = data_train[0]

print("Loading Model...")
net = MRnet()

image = image.view(1,-1,3,256,256)
image = image[:,:,:,:224,:224]

print(net(image))