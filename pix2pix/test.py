import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models import Generator
#from datasets.facade_dataset import Dataset
from datasets.facade_dataset_new import Dataset
from fid import fid

import numpy as np
import cv2
import statistics

batchsize =1
input_channel = 3
output_channel= 3
input_height = input_width = 256
fid_table = []
fid_control_table = []

input_data = Dataset(data_start = 300, data_end = 378)
test_len = input_data.len()
print(test_len)
generator_G = Generator(input_channel, output_channel).cuda()
generator_G.load_state_dict(torch.load('/content/gdrive/MyDrive/Colab_Notebooks/generator_G.pth'))

input_x_np = np.zeros((batchsize, input_channel, input_height, input_width)).astype(np.float32)

def tanh_range2uint_color(img):
    return (img * 128 + 128).astype(np.uint8)

def modelimg2cvimg(img):
    cvimg = np.array(img[0,:,:,:]).transpose(1,2,0)
    return tanh_range2uint_color(cvimg)

for iterate in range(test_len):
    image = input_data.get_image(iterate)
    input_x_np[0,:] = np.asarray(image[0])

    input_x = Variable(torch.from_numpy(input_x_np)).cuda()
    out_generator_G = generator_G.forward(input_x)

    out_gen = out_generator_G.cpu()
    out_gen = out_gen.data.numpy()
    cvimg = modelimg2cvimg(out_gen)
    calc_fid = fid()
    cv2.imwrite("./results/testGenImg%d.jpg"%iterate, cvimg)

    cvimg = modelimg2cvimg(input_x_np)
    cv2.imwrite("./results/testInputImg%d.jpg"%iterate, cvimg)

    img_test = cv2.imread("./results/testGenImg%d.jpg"%iterate, cv2.IMREAD_GRAYSCALE)
    img_in = cv2.imread("./results/testInputImg%d.jpg"%iterate, cv2.IMREAD_GRAYSCALE)
    fid_table.append(fid.calculate_fid(img_test, img_in))
    fid_control_table.append(fid.calculate_fid(img_in, img_in))

print('FID (control): Moyenne = ', statistics.mean(fid_control_table), ', Max = ', max(fid_control_table), ', Min = ', min(fid_control_table))
print('FID (test): Moyenne = ', statistics.mean(fid_table), ', Max = ', max(fid_table), ', Min = ', min(fid_table))
#print('FID (control): %.3f' % fid.calculate_fid(img_in, img_in))
#print('FID (test): %.3f' % fid.calculate_fid(img_test, img_in))

