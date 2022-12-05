
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
import cv2
import PIL
from PIL import ImageFile
from IPython.core.display import Image
from torchvision import datasets, models, transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pandas.core.base import NoNewAttributesMixin


class FundusDataset(torch.utils.data.Dataset):
    def __init__(self, image_path='.../train/',
                 VSI_path = '',
                 TDA_path='',
                 label_path='xxx.csv',
                 transform=None):
        """
        3D fundus dataset loader- 
        :param image_path: path to image folder
        :param VSI_path: path to mask folder
        :param TDA_path: path to mask folder
        :param label_path: path to labels file
        :param transform: Optional image transforms
        """

        self.transform = transform
        self.img_dir = image_path
        self.vsi_dir = VSI_path
        self.tda_dir = TDA_path 
        self.image_paths = sorted(os.listdir(self.img_dir))
        self.labels = pd.read_csv(label_path)

        ##### For Case Study 1 #####

    #     labels = np.load(label_path, allow_pickle=True).item()
    #     keys = list(labels['image_diagnoses'].keys())
    #     values = list(labels['image_diagnoses'].values())
    #     self.keys = []
    #     self.values = []
    #     for i in range(len(labels['image_diagnoses'])):
    #    #  if keys[i][0] == 'd' or keys[i][0] == 'a' or keys[i][0] == 'h':
    #       if keys[i][0] == 'a' or keys[i][0] == 'h':
    #    #  if keys[i][0] == 'h':
    #         self.keys.append(keys[i])
    #         self.values.append(values[i][0])


    def __len__(self):
       # return len(self.image_paths)
       return len(self.labels)

    def __getitem__(self, idx, plot=False):

        labels = self.labels.iloc[idx]
        img_name = labels['Filename']
        #img_name = sorted(self.image_paths)[idx]
        label = labels['Label'].astype(float)
        if label == 0:
          label = 0
        else:
          label = 1

        img = PIL.Image.open(os.path.join(self.img_dir, img_name))
        img = np.array(img)
       # img = np.tile(img, (3, 1))  if swithcing channels needed
       # img = Image.fromarray(img)

        vsi = PIL.Image.open(os.path.join(self.vsi_dir, img_name))
        vsi = np.array(vsi)

        tda_names = sorted(os.listdir(self.tda_dir))
        temp = [x for x in tda_names if 'VR' in x]
        tda_name = 'DS1_im' + temp[idx] +  '_persistence.png'  # for loading VR_filtration images
        tda = PIL.Image.open(tda_name)
        tda = np.array(tda)
        tda = tda[:,:,:-1]

        preprocess = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485], std=[0.229]),])

        if self.transform is not None:
            img = self.transform(img)
            vsi = self.transform(vsi)
            vsi = vsi[0, :, :]
            tda = Image.fromarray(tda)
            tda = self.transform(tda)

        else:
            img = self.preprocess(img)
            vsi = self.preprocess(vsi)
            vsi = vsi[0, :, :]
            tda = Image.fromarray(tda)
            tda = preprocess(tda)
            
        label = np.array(label)
        label = torch.from_numpy(label.copy()).float()

        return img, vsi, tda, label


def center_crop_or_pad(self, input_scan, desired_dimension):
        input_dimension = input_scan.shape
        #print('Input dimension: ', input_dimension, '\ndesired dimension: ', desired_dimension)

        x_lowerbound_target = int(np.floor((desired_dimension[0] - input_dimension[0]) / 2)) if desired_dimension[0] >= input_dimension[0] else 0
        y_lowerbound_target = int(np.floor((desired_dimension[1] - input_dimension[1]) / 2)) if desired_dimension[1] >= input_dimension[1] else 0

        x_upperbound_target = x_lowerbound_target + input_dimension[0] if desired_dimension[0] >= input_dimension[0] else None
        y_upperbound_target = y_lowerbound_target + input_dimension[1] if desired_dimension[1] >= input_dimension[1] else None


        x_lowerbound_input = 0 if desired_dimension[0] >= input_dimension[0] else int(np.floor((input_dimension[0] - desired_dimension[0]) / 2))
        y_lowerbound_input = 0 if desired_dimension[1] >= input_dimension[1] else int(np.floor((input_dimension[1] - desired_dimension[1]) / 2))

        x_upperbound_input = None if desired_dimension[0] >= input_dimension[0] else x_lowerbound_input + desired_dimension[0]
        y_upperbound_input = None if desired_dimension[1] >= input_dimension[1] else y_lowerbound_input + desired_dimension[1]


        output_scan = np.zeros(desired_dimension).astype(np.float32)  

        output_scan[x_lowerbound_target : x_upperbound_target, \
                    y_lowerbound_target : y_upperbound_target, ] 
        input_scan[x_lowerbound_input: x_upperbound_input, \
                   y_lowerbound_input: y_upperbound_input, ]

        return output_scan
