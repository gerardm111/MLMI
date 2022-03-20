from image_similarity_measures.quality_metrics import fsim, psnr, rmse
from sewar.full_ref import vifp, msssim, ssim
import fid
from sporco.metric import gmsd
import cv2

def similarity_computation(image_real_path, image_pred_path):
    img_real = cv2.imread(image_real_path)
    img_pred = cv2.imread(image_pred_path)
    if img_real.shape != img_pred.shape:
        img_pred = cv2.resize(img_pred[:2], img_real.shape[:2])
    img_real_gray = cv2.cvtColor(img_real, cv2.COLOR_BGR2GRAY)
    img_pred_gray = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)

    measures = []
    measures.append(fsim(img_real, img_pred))
    measures.append(ssim(img_real, img_pred)[0]) # index 1 corresponds to cs value
    measures.append(psnr(img_real, img_pred))
    measures.append(rmse(img_real, img_pred))
    measures.append(vifp(img_real, img_pred))

    temp = msssim(img_real, img_pred).real
    measures.append(temp)
    measures.append(gmsd(img_real_gray, img_pred_gray))
    measures.append(fid.calculate_fid(img_real_gray, img_pred_gray))

    return measures

print(similarity_computation("images/real184.png", "images/testGenImg184.jpg"))