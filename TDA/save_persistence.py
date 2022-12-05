
import numpy as np
import matplotlib.pyplot as plt
from persim import PersistenceImager
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='STARE', type=str)
parser.add_argument('--output', default='', type=str)
#parser.add_argument('--device', default=1, type=int)
args = parser.parse_args()


def largest(arr):
   # Initialize maximum element
   arr = arr.flatten()
   n = len(arr)
   mx = arr[0] 
   
   # Traverse array elements from second and compare every element with current max         
   for i in range(1, n):         
     if arr[i] > mx:         
         mx = arr[i]  
          
   return mx

dataset = args.dataset
output = args.output
#device = args.device

if dataset == 'STARE':

    label_path = '.../TDA/Case_study_1/Diagnoses/image_diagnoses_all.npy'
    labels = np.load(label_path, allow_pickle=True).item()
    keys = list(labels['image_diagnoses'].keys())
    values = list(labels['image_diagnoses'].values())
    stare_keys = [key for key in keys if key[0] == 'a']


    for key in stare_keys:
        key = key[1:]
        img = np.load('.../TDA/Case_study_1/Dataset_1_Results/DS1_im' + key + '_VR_persistence.npy', allow_pickle = True).item()
        test = img['BD']
        coordinates = np.array(test[1])
        pimgr = PersistenceImager(pixel_size=0.5)

        mx = largest(coordinates)
        coordinates = np.nan_to_num(coordinates, copy=True, posinf=mx)

        pimgr.fit(coordinates, skew=True)
        print(pimgr)
        print(pimgr.resolution)
        pimgs = pimgr.transform(coordinates, skew=True)

        fig, axs = plt.subplots(1, 3, figsize=(10,5))

        axs[0].set_title("Original Diagram")
        pimgr.plot_diagram(coordinates, skew=False, ax=axs[0])
        extent0 = axs[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output + '/Datasets/Persistence Images/STARE/' + key + '_diagram.png', bbox_inches=extent0)

        axs[1].set_title("Birth-Persistence\nCoordinates")
        pimgr.plot_diagram(coordinates, skew=True, ax=axs[1])
        extent1 = axs[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output + '/Persistence Images/STARE/' + key + '_coordinates.png', bbox_inches=extent1)

        axs[2].set_title('Persistence Image' + key + 'VR')
        pimgr.plot_image(pimgs, ax=axs[2])
        extent2 = axs[2].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(output + '/Persistence Images/STARE/'+ key + '_persistence.png', bbox_inches=extent2)

        plt.tight_layout()
        print(key, 'done')