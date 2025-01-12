import cv2 as cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import SimpleITK as sitk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import glob

import re

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



def affine_backward(slice, middle_image, now_idx, now_middle_idx, images_folder, affine_folder, output_folder):
    now_image = plt.imread(slice)[:,:,:3]

    affine_matrix = np.zeros((3,3))
    affine_matrix[2,2] = 1.

    for sgi in range(now_idx, now_middle_idx):
        if sgi == now_idx:
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            affine_matrix[:2,:] = cv2.invertAffineTransform(np.load(matrix_root))
            print(matrix_root)
        else:
            matrix_root = os.path.join(affine_folder, '%s-to-%s' % (sgi + 1, sgi), 'sg_affine_init.npy')
            new_affine = np.zeros((3,3))
            new_affine[2,2] = 1.
            new_affine[:2, :] = cv2.invertAffineTransform(np.load(matrix_root))
            affine_matrix = affine_matrix.dot(new_affine)
            print(matrix_root)

    affine_matrix_inv = cv2.invertAffineTransform(affine_matrix[:2,:])

    img1_affine = cv2.warpAffine(now_image, affine_matrix[:2,:], (middle_image.shape[1], middle_image.shape[0]))
    new_root = slice.replace(images_folder, output_folder)
    plt.imsave(new_root, img1_affine)


def affine_IHCtoPAS(slice, middle_image, images_folder, affine_folder, output_folder):
    now_image = plt.imread(slice)[:, :, :3]

    affine_matrix = np.zeros((3,3))
    affine_matrix[2,2] = 1.

    matrix_root = os.path.join(affine_folder, 'IHC-to-PAS', 'sg_affine_init.npy')
    affine_matrix[:2, :] = np.load(matrix_root)

    img1_affine = cv2.warpAffine(now_image, affine_matrix[:2,:], (middle_image.shape[1], middle_image.shape[0]))
    new_root = slice.replace(images_folder, output_folder)
    plt.imsave(new_root.replace('.png','affine.png'), img1_affine)

if __name__ == "__main__":

    case_folder = '/Data/MolecularEL/PAS+Endo_png'
    cases = glob.glob(os.path.join(case_folder, '*'))
    cases.sort(key=natural_keys)
    now_middle_idx = 1

    for ci in range(len(cases)):
    #for ci in range(0,1):
        case = cases[ci]
        now_case = os.path.basename(case)

        images_folder = os.path.join(case, '5X')
        affine_folder = os.path.join(case, 'sg_affine')
        output_folder = os.path.join(case, 'affine_5X')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        image = glob.glob(os.path.join(images_folder, '*IHC.png'))[0]
        middle_image = plt.imread(image.replace('IHC.png','PAS.png'))

        affine_IHCtoPAS(image, middle_image, images_folder, affine_folder, output_folder)
    
        plt.imsave(image.replace('IHC.png','PAS.png').replace(images_folder, output_folder), middle_image)




