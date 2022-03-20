import glob
import yaml
import os
import time
import statistics

path = r'C:\Users\mahau\Desktop\Télécom\3A_S1\PRAKTIKUM-MLMI\Implementation\BraTS20_Training\*'+"/flair/*"
real_paths = glob.glob(path)
print('test1: path: ', len(real_paths), real_paths)

with open('indices_test.yml') as file:
    indices = yaml.safe_load(file)
#print('test2: yaml loading: \n', indices)
#print(indices['val'])

root = r"C:\Users\mahau\Downloads\brats_data\data\brats20\train"
flair_dirs = [os.path.join(root+'/', x.replace("*", "flair")) for x in indices['train']]
#print('test3: selection indices: \n', flair_dirs)

list = []
for i in range (10):
    list.append(i)
    print("loaded: ", i, " ", end="\r")
    time.sleep(0.5)
print("all loaded!")
print(min(list), ', ', max(list),', ', statistics.mean(list))