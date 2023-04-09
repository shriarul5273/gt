
import os
from glob import iglob
from PIL import Image
import json
import pandas as pd
from torch.utils.data import Dataset,DataLoader,random_split
from torchvision import transforms
import numpy as np
import torch

category_dict = pd.read_pickle('cat_dict.pkl')


class BEHAVE(Dataset):
    def __init__(self, imagePaths, transform=None):
        self.imagePaths = imagePaths
        self.transform = transform

    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, index):
        imagePath = self.imagePaths[index]
        image = Image.open(imagePath[0])
        csv = pd.read_pickle(imagePath[1])
        a = torch.tensor(np.array([csv['angle'],csv['trans']]), dtype=torch.float32)
        y = torch.zeros((20,2,3), dtype=torch.float32)
        y[category_dict[imagePath[2]],:,:] = a
        if self.transform:
            image = self.transform(image)
        return image, y, imagePath[2]




def get_dataloader():
    imagePaths = []
    paths = iglob('/kaggle/input/behave/*')
    # paths = iglob('/home/shriarul/Downloads/Chore/train_part1/*')
    paths = [x for x in paths if os.path.isdir(x)]
    category  = []
    for path in paths:
        info = json.load(open(os.path.join(path, 'info.json')))
        cat = info['cat']
        if cat not in category:
            category.append(cat)
        subPaths = iglob(os.path.join(path, '*'))
        subPaths = [x for x in subPaths if os.path.isdir(x)]
        for subPath in subPaths:
          subsubPaths = iglob(os.path.join(subPath, '*'))
          subsubPaths = [x for x in subsubPaths if os.path.isdir(x)] 
          objects = []
          for k in range(len(subsubPaths)):
              if 'person' not in subsubPaths[k]:
                objects.append(subsubPaths[k])
          for i in info['kinects']:
              for h in range(len(objects)):
                object_name = os.path.basename(objects[0])
                xpath = os.path.join(subPath, 'k{}.color.jpg'.format(i))
                ypath = os.path.join(subPath,object_name, 'fit01', '{}_fit.pkl'.format(object_name))
                imagePaths.append((xpath, ypath,cat))

    # print(len(imagePaths))
    # print('+++++++++++++++++++++++')
    # for i,imagePath in enumerate(imagePaths):
    #     image = Image.open(imagePath[0])
    #     print(imagePath[0])
    #     csv = pd.read_pickle(imagePath[1])
    #     print(csv)

    dataset = BEHAVE(imagePaths, transform=transforms.Compose([transforms.Resize((384,512)),transforms.ToTensor()]))
    #dataset = BEHAVE(imagePaths, transform=transforms.ToTensor())
    # dataset = BEHAVE(imagePaths, transform=transforms.Compose([transforms.ToTensor()]))


    # trainDataset, valDataset = random_split(dataset, [12612,100 ])
    # print(len(dataset))
    trainDataset, valDataset = random_split(dataset, [10712, 2000])


    # trainLoader = DataLoader(trainDataset, batch_size=1, shuffle=True, num_workers=4)
    # valLoader = DataLoader(valDataset, batch_size=1, shuffle=True, num_workers=4)

    
    trainLoader = DataLoader(trainDataset, batch_size=20, shuffle=True, num_workers=2)
    valLoader = DataLoader(valDataset, batch_size=20, shuffle=True, num_workers=2)

    return trainLoader, valLoader

# dataloaders = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
# print(len(dataloaders))

# d = next(iter(dataloaders))

# print(d[0].shape)
# print(d[1].shape)
# print(d[2])


dataloaders, _ = get_dataloader()


# print(len(dataloaders))

d = next(iter(dataloaders))

# print(d[0].shape)
print(d[1].shape)
# print(d[2])
