---
layout: default
title: 'methods'
---

### Methods 

#### Class Balancing
Upsampling was done by sampling background spectra, the minority class, with replacement until the number of background spectra matched the number of foreground spectra in the training, testing, and validation sets. Downsampling was done by random selecting foreground spectra, the majority class, without replacement until the classes were balanced. Sklearn's resampling function was used for both of these class balancing methods.


#### Classification
Three basic models were first used to classify Raman spectra from individual pixels as either nuclei (foreground) or background: Logistic Regression, Random Forests, and a shallow Neural Net with three fully-connected hidden layers, each followed by ReLu activation and Dropout layers. The output layer was also fully-connected with two activation units. 

Logistic Regression and Random Forests were implemented in Python's sklearn using default parameters. The Neural Net was created in Tensorflow and Keras. Optimal Hyperparameters were chosen using Random Search


#### Patch-CNN
I created Patch-CNN, a shallow networks consisting of the following layers: a Convolution Layer with kernel size 2 and 64 activation units, a Max Pooling Layer, two fully-connected hidden layer -- each followed by a Dropout Layer -- and one fully-connected output layer.   Hyperparameters including learning rate, dropout rate, L2 regularization constant,  number of activation units and activation functions were chosen using an optimized version of Random Search, implemented using keras-tuner. 

Prior to training Patch-CNN, I created the input using Raman tensor patches.  Rather than using the Raman spectra from one pixel alone, I created small tensors of size (3, 3, raman-depth) consisting of a Raman spectra from one pixel and its eight surrounding pixels. If there weren't eight pixels surrounding a given Raman spectra, I would impute any missing spectra with the mean of available surrounding spectra. Each Raman tensor patch had at least four available spectra to use for imputing while a large majority had nine spectra. 

#### Training and Testing
The training set was made by randomly sampling 60\% of the images. The remaining 40\% of images were divided equally into validation and test sets. The same training, validation, and test sets were used for training and evaluating all four models used in this project. \\

The neural networks were trained using each training set to maximize validation accuracy and area under the ROC curve using the validation set. Adam was used as the optimizer and Binary Cross Entropy as the loss function in both the neural networks\\


#### Model Evaluation
I evaluated each model on the test set consisting of portions of images that had reliable labels. I generated an ROC curve for each model's performance on this smaller test set.  Additionally, I predicted labels (foreground or background) for every pixel in randomly-chosen test images and displayed them next to an image taken of the same spatial location (nuclei stain image) and a state-of-the-art segmentation from NucleAIzer. These representations are shown in Figures \ref{mapped_img_up}, \ref{mapped_img_down}, and \ref{mapped_img_patch_cnn}. 

#### Mapping Predictions to an Image
Because the outputs of the network were taken from softmax activation, I generated a probability that each spectra belonged to foreground and background. For each spectra (i.e. pixel), I took the probability that it belonged to the foreground and generated an image using these probabilities. The inputs to the model were ordered spatially, making it only necessary to reshape the output probabilities to generate an image. 
