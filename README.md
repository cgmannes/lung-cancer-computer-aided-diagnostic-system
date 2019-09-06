# Lung Cancer Computer-Aided Diagnostic System

This respository contains python files for a computer-aided diagnostic (CAD) system for lung nodule classification in 
magnetic resonance images. The CAD system employs two different approaches for nodule classification. 

The first method employs Local Binary Patterns (LBP) enhanced with Histogram of Oriented Gradients (HOG) to construct
a feature vector. Classification is achieved using K-Nearest Neighbors technqique. The second approach uses a VGG 16
Convolutional Neural Network (CNN) with pretrained weights for nodule classification. 

cancerCAD.py: This files contains code for importing the images, and applying any necessary preprocessing operations.
Then, KNN classification is performed.

noduleClassifier.py: This file imports images pre-process by cancerCAD.py and performs subsequent pre-processing
techniques for CNN and then trains a VGG 16 CNN.

CAD_poster.pdf : Provides an overview of the work and the results of the project.