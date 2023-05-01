import time
import numpy as np
import torch
import cv2
from regressCNN import RegressionPCA
from widgets_HSE import getBinaryimage, getSamplePoints, save_obj, repeat_data   
from .obj_utils.io import *

def reconstruction_vertices(base, mean, coeff, height):
    X_reconstructed = coeff.dot(base) + mean.reshape((-1))
    X_reconstructed = X_reconstructed.reshape((-1,6449 ,3))
    X_reconstructed *= height

    return X_reconstructed

def main(name, input_frontal, input_lateral, height):
    name = name
    body_height = height
    img_filenames = [input_frontal, input_lateral]

    outbody_filenames = './Images/{}.obj'.format(name)

    pcaBase = np.load('./infer/caesar_pca_base.npy')
    mean = np.load('./infer/caesar_mean.npy')

    mesh_faces = np.load('./infer/face.npy')

    # load input data
    sampling_num = 648
    data = np.zeros([2, 2, sampling_num])
    for i in np.arange(len(img_filenames)):
        img = img_filenames[i]
        im = getBinaryimage(img, 600) # deal with white-black image simply
        sample_points = getSamplePoints(im, sampling_num, i) 
        center_p = np.mean(sample_points, axis = 0)
        sample_points = sample_points - center_p
        data[i, :, :] = sample_points.T

    data = repeat_data(data)

    # load CNN model
    print('==> begining...')
    len_out = 22
    model_name = './Models/model_male_50.ckpt'
    ourModel = RegressionPCA(len_out)
    ourModel.load_state_dict(torch.load(model_name))
    ourModel.eval()

    # output results
    coeff = ourModel()

    mesh_vertices = reconstruction_vertices(pcaBase, mean, coeff,body_height)
    
    save_obj('output.obj',mesh_vertices,mesh_faces)
    # save_obj(outbody_filenames, ourModel, body_height, data)



if __name__ == '__main__':
    main()

    # img_filenames = ['./Images/5front.jpg', './Images/6side.jpg']
    # img = cv2.imread(img_filenames[1])
    # img = cv2.flip(img,1)
    # cv2.imwrite('6side1.jpg',img)
