# THis is file is only for debug purposes

import torch
from models import MRnet
from dataset import MRData

# print("Loading Data...")
# data_train = MRData(plane='axial',task='acl')

# train_loader = torch.utils.data.DataLoader(
#         data_train, batch_size=1, shuffle=True, num_workers=11, drop_last=False)

# for x,y in train_loader:
#     print(x[1].shape)
#     print(y.shape)
#     break

# image,label = data_train[0]

print("Loading Model...")
net = MRnet()

# print(net)

image = torch.rand(20,3,224,224)
image = image.view(1,-1,3,224,224)

images = [image,image,image]

print(net.forward(images))