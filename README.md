# Goal

Medical datasets often have a limited number of samples with anomalies, which is problematic to train machine learning algorithms. Machine Learning algorithms for segmentation tasks can be usefull for disease detection in the medical area. Synthetically generated anomalous data via inpainting techniques can then improve the performance of segmentation methods.

Have a pipeline/method that can generate anomalies via inpainting technique in different modalities (brain, lung, liver). First, test and trials will only consist brain tumors. If everything goes well other modalities will be tested as well.

# Speckle
**Removal:** Using filters (crimmins, mean, median). Remove this source of noise during pre-processing, before the GAN. [Filters](preprocessing/mlmi-contour.ipynb), [4]

**Generation:** Add speckle to synthetic images, output of the GAN. The speckle transformation was implemented according to the Mathematica implementation of [5]. [Speckle in Python](speckle.ipynb) , [Speckle in Mathematica](speckle_mathematica.pdf)

The idea is to spare GANs from learning parameters linked to the speckle noise observed in real medical images (CT, ultrasound but NOT MRI). 

# Datasets

### BraTS2018 & BraTS2020

3D MRI: overhead slices where done to have 2D images to work with.

Data pre-processing done to have as much as possible homogeneous 2D images.

Brain tumor segmentation [link][10]
There are four multimodal scans in the dataset:
1. Native (T1)
2. Post-contrast T1-weighted (T1Gd)
3. T2-weighted (T2)
4. T2 Fluid Attenuated Inversion Recovery (FLAIR) volumes

[MRI Basics][13]

There are three sub-regions of a "Glioma" brain tumor segmentation for evaluation:
1. the "enhancing tumor" (ET)
2. the "tumor core" (TC)
3. the "wholetumor" (WT)

### Dataset with healthy samples
https://drive.google.com/drive/folders/1kMHEf94rdGy8E1vgtZBF4_klwtY1ugbU?usp=sharing

trainA & testA contain healthy brains / trainB & testB contain tumor brains

This dataset was built using https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset 

# Data pre-processing

Check [kaggle][15] or [mlmi-contour.ipynb](preprocessing/mlmi-contour.ipynb) for:
- image filtering
- image contour
- image pre-processing
- data augmentation
- inference test from brain tumor paper [2]

The tests of this section were done using Brain tumor MRI dataset [data][14].

# Evaluation metrics
- Fréchet Inception Distance (FID) in [**0**, +inf]

### Image Quality metrics [11]
- Feature-based similarity index (FSIM) [7], in [0, **1**]
- Structural similarity measure (SSIM) [8], in [0, **1**]
- Peak signal-to-noise ration (PSNR) [7], in [0, **inf**]
- Root mean square error (RMSE) [7], in [**0**, 1]
- Visual information fidelity (VIF) [8], in [0, **1**]
- Multi-scale SSIM (MSSSIM) [8], in [0, **1**]
- Gradient magnitude similarity deviation (GMSD) [9], in [**0**, 1]

# Pix2pix
Based on [16] training of pix2pix models with several parameter settings to go from segmentation map to MRI image.

# Work schedule
Week | Tasks | Associated code
------- | ------ | ------- 
20/10-27/10 | Litterature review on inpainting techniques and anomaly generation  | [References](#References) 
28/10-10/11 | Implementation of a simple GAN for synthetic generation of mnist digits, working. Litterature review on inpainting techniques and generative models | [Simple GAN](GAN_on_MNIST_digits.ipynb)
11/11-24/11 | Familiarization with speckles: generation and suppression (filters: mean, median, crimmins). Litterature review on speckles and SPADE. Data pre-processing, including data augmentation, contour, mask pre-processing. Test inferences using pre-trained weights of [6] | [Data Preprocessing](preprocessing/mlmi-contour.ipynb)
25/11-01/12 | Speckle generation theory (contact with author from [3], work on the appendix of book [5], familiarization with the mathematica language) | [Speckle in Python](speckle.ipynb) , [Speckle in Mathematica](speckle_mathematica.pdf)
02/12-08/12 | Familiarization with the Brats dataset, pre-processing (from 3D MRI to 2D slices and homogeneisation of the dataset) | [BraTS Preprocessing](preprocessing/load_BRATS_data.ipynb)
09/12-15/12 | Tentative of VM deployment, problems on Quotas remaining. Presentation work (Motivation, Speckles, datasets, general structure). | 
16/12-05/01 | Pix2pix from tumor segmentation to MRI running on colab, adding a GPU execution to some parts of the code (fasten the execution). Training and testing for different numbers of epochs, and visual comparison. [16] | [Pix2pix](pix2pix)
06/01-19/01 | Adding a speckle layer to the pix2pix generator and backpropagation: technique isn't working | [Pix2pix](pix2pix)
20/01-26/01 | Implementation of the Frechet Inception Distance and evaluation of the pix2pix model with it (on the testing set, mean, maximum and minimum values are observed). Training and testing dataset using BraTS20 and the common set of indices (25% of the data). | [FID](pix2pix/fid.py) , [Indexes](pix2pix/brats20_25p.yml) , [Data Loader](pix2pix/datasets/facade_dataset_new.py)
27/01-02/02 | Pix2pix (from segmentation to MRI) training with different loss functions and also parameters (lambda, learning rate, optimizers), limited to 10 epochs and 25% of BraTS20 dataset | [Training](pix2pix/train.py) 
03/02-09/02 | Creation of a dataset for CycleGAN (healthy brains and tumor brains into train & test folders). Training of CycleGAN to go from healthy brains to tumor brains and vice-versa [12] | [CycleGAN](CycleGAN), [Dataset](https://drive.google.com/drive/folders/1kMHEf94rdGy8E1vgtZBF4_klwtY1ugbU?usp=sharing)
10/02-20/02 | Similarity measure implementation to evaluate our different models [17] | [Metrics](similarity_evaluation.py)

# References
- [Semantic Image Synthesis with Spatially-Adaptive Normalization][1]
- [Synthesis of Brain Tumor MR Images for Learning Data Augmentation][2]
- [SpeckleGAN: a generative adversarial network with an adaptive speckle layer to augment limited training data for ultrasound image processing][3]
- [Removal Speckle Noise from Medical Image Using Image Processing Techniques][4]
- [Speckle phenomena in optics, theory and applications][5]
- [Generative Image Inpainting with Contextual Attention][6]
- [Comparison of Objective Image Quality Metrics to Expert Radiologists’ Scoring of Diagnostic Quality of MR Images][11]
- [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks][12]
- [Image-to-Image Translation with Conditional Adversarial Networks][16]
- [Comparison of Objective Image Quality Metrics to Expert Radiologists' Scoring of Diagnostic Quality of MR Images][17]

### Other ressources
- [Image-simiarity-measures module][7]
- [Sewar module][8]
- [Sporco.metric module][9]
- [BraTS2018][10]
- [MRI Basics][13]
- [Brain Tumor MRI dataset][14]
- [Mlmi-contour kaggle code][15]


[1]: <https://arxiv.org/abs/1903.07291> (Semantic Image Synthesis with Spatially-Adaptive Normalization)
[2]: <https://arxiv.org/abs/2003.07526> (Synthesis of Brain Tumor MR Images for Learning Data Augmentation)
[3]: <https://www.researchgate.net/publication/342288578_SpeckleGAN_a_generative_adversarial_network_with_an_adaptive_speckle_layer_to_augment_limited_training_data_for_ultrasound_image_processing> (SpeckleGAN: a generative adversarial network with an adaptive speckle layer to augment limited training data for ultrasound image processing)
[4]: <http://ijcsit.com/docs/Volume%207/vol7issue1/ijcsit2016070183.pdf> (Removal Speckle Noise from Medical Image Using Image Processing Techniques)
[5]: <https://www.google.de/books/edition/Speckle_Phenomena_in_Optics/TynXEcS0DncC?hl=en&gbpv=1&printsec=frontcover> (Speckle phenomena in optics, theory and applications)
[6]: <https://openaccess.thecvf.com/content_cvpr_2018/papers/Yu_Generative_Image_Inpainting_CVPR_2018_paper.pdf> (Generative Image Inpainting with Contextual Attention)
[7]: <https://pypi.org/project/image-similarity-measures/> (image-simiarity-measures module)
[8]: <https://pypi.org/project/sewar/> (sewar module)
[9]: <https://sporco.readthedocs.io/en/v0.1.9/sporco.metric.html> (sporco.metric module)
[10]: <https://arxiv.org/abs/1811.02629> (BraTS2018)
[11]: <https://ieeexplore.ieee.org/document/8839547> (Comparison of Objective Image Quality Metrics to Expert Radiologists’ Scoring of Diagnostic Quality of MR Images)
[12]: <https://arxiv.org/pdf/1703.10593.pdf> (Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks)
[13]: <https://case.edu/med/neurology/NR/MRI%20Basics.htm> (MRI Basics)
[14]: <https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset> (Brain Tumor MRI dataset)
[15]: <https://www.kaggle.com/mahautgrard/mlmi-contour> (My Kaggle code)
[16]: <https://arxiv.org/pdf/1611.07004.pdf> (Image-to-Image Translation with Conditional Adversarial Networks)
[17]: <https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8839547> (Comparison of Objective Image Quality Metrics to Expert Radiologists' Scoring of Diagnostic Quality of MR Images)
