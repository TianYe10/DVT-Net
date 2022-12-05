import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.transforms import RandomAutocontrast
from fundusdataset import FundusDataset

''' 
Define transforms
'''

preprocess_train = transforms.Compose([
   # transforms.RandomResizedCrop(224),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
   # transforms.CenterCrop(224),
   # transforms.RandomHorizontalFlip(p = 0.5),
   # transforms.RandomRotation(degrees=(-90, 90)),
   # transforms.RandomAutocontrast(p = 0.5),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

### other options oftransformations should be loaded accordingly ##
preprocess_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


'''
Create (Pytorch) Datagenerators
'''

def image_dataloader(train_image_path, train_VSI_path, train_TDA_path, train_label_path = 'train_label.csv', 
                   bs = 5, 
                   val_image_path = '', val_VSI_path = '', val_TDA_path = '', val_label_path = 'val_label.csv' ):
            
    image_datasets = {
        'validation': 
        FundusDataset(
                    image_path = train_image_path,
                    VSI_path = train_VSI_path,
                    TDA_path= train_TDA_path,
                    label_path = train_label_path,
                    transform=preprocess_val),
        
        'train': FundusDataset(
                    image_path = val_image_path,
                    VSI_path = val_VSI_path,
                    TDA_path= val_TDA_path,
                    label_path = val_label_path,
                    transform=preprocess_train)
    }

    dataloaders = {
        'train':
        torch.utils.data.DataLoader(image_datasets['train'],
                                    batch_size=bs,
                                    shuffle=True),
        'validation':
        torch.utils.data.DataLoader(image_datasets['validation'],
                                    batch_size=bs,
                                    shuffle=False)
    }

    print('image_datasets, dataloaders ready')

    return image_datasets, dataloaders

