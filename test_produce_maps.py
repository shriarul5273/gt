import torch
import os
from Code.lib.model import SPNet
from PIL import Image
from glob import iglob
import json
from torchvision import transforms
import pandas as pd
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle as pkl
import numpy as np
#load the model
model = SPNet(32,50)
model.cuda()

model.load_state_dict(torch.load('/home/shriarul/Downloads/BEHAVE_Object_detection/SPNet/Checkpoint/SPNet/SPNet_epoch_best.pth'))
model.eval()


category_dict = pd.read_pickle('cat_dict.pkl')

def get_testdataloader():
    imagePaths = []
    # paths = iglob('/kaggle/input/behave/*')
    paths = iglob('/home/shriarul/Downloads/BEHAVE_Object_detection/test_images/*')
    paths = [x for x in paths if os.path.isdir(x)]    
    for path in paths:
        info = json.load(open(os.path.join(path, 'info.json')))
        cat = info['cat']
        gender = info['gender']
        subPaths = iglob(os.path.join(path, '*'))
        subPaths = [x for x in subPaths if os.path.isdir(x)]
        for subPath in subPaths:
            xpath = os.path.join(subPath, 'k1.color.jpg')
            imagePaths.append((xpath,cat,gender))

    return imagePaths


imagePaths = get_testdataloader()


# for i,imagePath in enumerate(imagePaths):
#     image = Image.open(imagePath[0])
#     print(imagePath[0])

transform = transforms.Compose([transforms.Resize((384,512)),transforms.ToTensor()])


k = {}

for imagePath in tqdm(imagePaths):
    image = Image.open(imagePath[0])
    image = transform(image)
    image = image.unsqueeze(dim=0).cuda()
    output = model(image)
    j = os.path.normpath(imagePath[0]).split(os.sep)
    rotation = R.from_rotvec(output[0,category_dict[imagePath[1]],0,:].cpu().detach().numpy()).as_matrix()
    translation = output[0,category_dict[imagePath[1]],1,:].cpu().detach().numpy()
    if j[-3] not in k.keys():
        k[j[-3]] = {}
        k[j[-3]]['obj_rots'] = np.array([rotation])
        k[j[-3]]['obj_trans'] = np.array([translation])
        k[j[-3]]['obj_scales'] = np.array([1])
        k[j[-3]]['frames'] = [j[-2]]
        k[j[-3]]['gender'] = imagePath[2]
    
    else:
        k[j[-3]]['obj_rots']  = np.vstack((k[j[-3]]['obj_rots'],np.array([rotation])))
        k[j[-3]]['obj_trans'] = np.vstack((k[j[-3]]['obj_trans'],np.array([translation])))
        k[j[-3]]['obj_scales'] = np.append(k[j[-3]]['obj_scales'],1)
        k[j[-3]]['frames'].append(j[-2])

    
pd.to_pickle(k,'test.pkl')

pkl.dump(k, open('test_pk.pkl', 'wb'))