import cv2
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\mahau\Downloads\18 (1).png")
print(img[0][0])
for i in range(len(img)):
    for j in range(len(img[i])):
        for k in range(len(img[i][j])):
            if img[i][j][k]==1:
                img[i][j][k] = 100
            elif img[i][j][k] == 2:
                img[i][j][k] = 255
                print('2')

plt.imsave(r'C:\Users\mahau\Desktop\Télécom\3A_S1\PRAKTIKUM-MLMI\EXEMPLES\ex 7\seg518.png', img)
plt.imshow(img)
plt.show()