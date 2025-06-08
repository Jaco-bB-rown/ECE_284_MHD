# ECE 284 Final Project

I am recreating " Melanoma diagnosis using deep learning techniques on dermatoscopic images".  

The Mask R-CNN model is trained and evaluated in "Mask_RCNN.ipynp".  

The ResNet-152 classification model is trained and evaluated in "ResNet_classification.ipynb".  

The dataset classes that I call in the Jupyter notebooks are held in "dataloader.py".  

The data augmentations I perform are made into csv files for the dataset class to use. These data augmentation csv files are made in "aug_csv_gen.ipynb".  

To crop the images for classification I generate csv files containing the predicated bounding box for each image. The class dataset uses these to crop the image. These csv files are generated with the Mask R-CNN model in "BB_Creation.ipynb".  

The datasets are stored in the folders "Dataset_2017" for the base ISIC 2017 dataset, "Dataset_2017_ph2" for the one containing ph2 images, and "Dataset_2018" for the ISIC 2018 dataset. These folders then contain a folder for images, masks, ground truth csv files, and preicated bounding box csv files. The image and mask folders are split into Training, Validation, and Test folders. All csv files follow the data convention of test, train, or validation followed by "GroundTruth" and then the data augmentation number (the numbers are explained in the paper).  

Datasets 2017_ph2 and 2018 follow largely the same naming conventions. For the 2018 dataset, data augmentation 7's csv is listed as 1 for compatibility reasons.  

The Results folder holds all the images, plots, and tables generated from testing my models. The confusion matrices have an "on_Dataset_" to indicate which dataset they were tested on. Latex Tables and the ROC scatter plot were generated in "results.ipynb".  

The original paper I am recreating is titled "Original_Project_Paper.pdf" here. The paper I wrote regarding my project is titled "ECE_284_Final_Project_Report.pdf" here.  

The "runs" folder holds various metrics obtained during training such as loss curves but I did not use any in my final project and it is too large to upload to GitHub. The "Trained_Models" folder holds every model I trained but each model is too big to upload to GitHub so if you would like them you can reach out to me.  