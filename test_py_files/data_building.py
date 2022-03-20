import cv2
import imutils
import os
import glob

if __name__ == "__main__":
    #training = r"C:\Users\mahau\Downloads\brats_data\data\brats20\train\lgg"
    testing = r"C:\Users\mahau\Downloads\brats_data\data\brats20\train\hgg\*\flair"
    #training_dir = os.listdir(training)
    testing_dir = glob.glob(testing)
    IMG_SIZE = 256
    cpt = 0
    cpt_dir = 1110

    for dir in testing_dir:
        save_path = '../HealthyBrain2TumorBrain/Training/glioma2'
        path = os.path.join(testing,dir)
        print(path)
        image_dir = os.listdir(path)
        cpt_dir += 100
        for img in image_dir:
            if cpt<1000:
                image = cv2.imread(os.path.join(path,img))
                new_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                new_img = cv2.resize(new_img,(IMG_SIZE,IMG_SIZE))
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv2.imwrite(save_path+'/'+str(cpt_dir)+img, new_img)
                cpt = cpt + 1
