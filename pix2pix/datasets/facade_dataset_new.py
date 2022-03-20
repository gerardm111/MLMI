import numpy as np
import cv2
import random
import glob
import os
import yaml

class Dataset():
    def __init__(self, dataDir='./brats20/', data_start = 1, data_end = 300, train = False):

        #train = False
        with open('./brats20_25p.yml') as file:
            indices = yaml.safe_load(file)
            print("yaml file loaded \n")

        self.dataset = []
        self.shufflelist = []

        if train:

          real_paths = [os.path.join(dataDir+'train/', x.replace("*", "flair")) for x in indices['train']]
          #real_path = dataDir + "train/hgg/BraTS20_Training_002"+"/flair/*"
          #real_paths = glob.glob(real_path)

          mask_path = [os.path.join(dataDir+'train/', x.replace("*", "seg")) for x in indices['train']]
          #mask_path = dataDir + "train/hgg/BraTS20_Training_002"+"/seg/"

          print("list of indices built \n")

        else:
          # For inference
          #real_path1 = dataDir + "validation/hgg/BraTS20_Training_001"+"/flair/42.png"
          real_paths = [os.path.join(dataDir+'validation/', x.replace("*", "flair")) for x in indices['val']]
          #real_paths = []
          #real_paths.append(real_path1)

          # mask_path = dataDir + "validation/hgg/BraTS20_Training_001"+"/seg/42.png"
          mask_path = [os.path.join(dataDir+'validation/', x.replace("*", "seg")) for x in indices['val']]

        for i in range(len(real_paths)):
            #print(len(real_paths), "(263): ", real_paths[i])
            real_image_path = real_paths[i]
            real_image_name = os.path.basename(real_image_path)

            real_image = cv2.imread(real_image_path)
            #print(real_image_path)
            real_image = self.resize(real_image, ipl_alg = cv2.INTER_CUBIC)                
            
            #mask_image_path = mask_path + real_image_name.replace("flair", "flair_seg_masked")
            mask_image_path = mask_path[i]
            #print(mask_image_path)
            mask_image = cv2.imread(mask_image_path)
            mask_image = self.resize(mask_image, ipl_alg = cv2.INTER_NEAREST)

            self.dataset.append((mask_image, real_image))

            self.shufflelist = list(range(self.len()))
            #print('image ',i, end="\r", flush=True)

        print("Dataset initialized with ", str(len(self.dataset)), " pairs of images")


    def resize (self, img, base_size = 286, ipl_alg = cv2.INTER_CUBIC):
        img_height, img_width, _ = img.shape
        short_side = min(img_height, img_width)
        rasio = base_size / short_side

        new_width  = int(img_width * rasio)
        new_height = int(img_height * rasio)

        return cv2.resize(img, (new_height, new_width), interpolation=ipl_alg)

    def uint_color2tanh_range(self, img):
        return img / 128. - 1.

    def len(self):
        return len(self.dataset)

    def shuffle(self):
        random.shuffle(self.shufflelist)

    def get_image(self, i):
        input_x_image = np.asarray(self.dataset[self.shufflelist[i]][0])
        real_image = np.asarray(self.dataset[self.shufflelist[i]][1])

        input_x_image, real_image = self.crip_imgs_pair(input_x_image, real_image)

        if random.random() > 0.5:
            input_x_image = cv2.flip(input_x_image, 1)
            real_image = cv2.flip(real_image, 1)        

        input_x_image = self.uint_color2tanh_range(input_x_image.astype(np.float32))
        real_image = self.uint_color2tanh_range(real_image.astype(np.float32))

        return input_x_image.transpose(2,0,1), real_image.transpose(2,0,1)

    def crip_imgs_pair(self, img1, img2, crip_size = 256):
        img_height, img_width, _ = img1.shape
        crip_x = random.randint(0, img_width - crip_size)
        crip_y = random.randint(0, img_height - crip_size)

        return img1[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :], img2[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :]

    def crip_img(self, img, crip_size = 256): 
        img_height, img_width, _ = img.shape
        crip_x = random.randint(0, img_width - crip_size)
        crip_y = random.randint(0, img_height - crip_size)

        return img[crip_y : crip_size + crip_y, crip_x : crip_size + crip_x, :]
