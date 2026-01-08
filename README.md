# RNSAPneumoniaDetectionPipeline

In this project I worked with the RSNA Pneumonia dataset from https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge.
This project went through many iterations but finally settled on
- a pipelie that first takes in a chest X-Ray
- the X-ray image is passed on to an ensemble of ViT and Resnet 101 classifiers
- if the classifiers predict "lung_opacit", the image is passed on to a Faster R-CNN model that detects lung opacity

The classifier models were trained to maximize macro recall. 
For the ViT mode, the macro recall is about 0.74.
For the Resnet 101 model, the macro recall is lower than that of the ViT model. (I actually forgot to note this value down. LEARNING: Don't trust kaggle notebooks to save training logs in output cells)
For the Faster R-CNN model, the training loss was about 0.2722.

What each file contains:
- driver.ipynb: Initial trials for building a model where I started off with creating a resnet 50 model that trained on both x-ray images as well as metadata. I intended to use this as a backbone for the Faster RCNN model. That plan failed. So I trained Faster RCNN model seperately only for the images where there was lung opacity.
- detector.ipynb: This notebook contains the training of my Faster RCNN model. According to my new plan, images will be passed to this model only if lung opacity is detected by classification models.
- vit_and_resnet_101_classifier.ipynb: contains the code for training ViT and resnet-101 classifiers.
- app.py: This contains code for the streamlit application.
