---
layout: default
title: "Overview"
---


### Overview
Raman spectroscopy has been shown to be a non-destructive, label-free method for determining molecular composition.  A Raman spectra can be taken of each pixel in a microscopic image, representing a high-dimensional spatial tensor that may be useful in extracting meaningful biological signal.  In this project, I will explore the use of Spatial Raman spectra in segmenting images of nuclei, distinguishing individual cells from their backgrounds. I compare several pre-processing methods and standard classifiers in distinguishing foreground from background as well as map foreground probability predictions to a 2D image, generating a segmentation map. Additionally, I will create and test a convolutional neural network based on Raman tensor patches for image segmentation.



<img src="images/mapped_img_patch_cnn.png">



### Motivation
Image  segmentation  is  a  common  problem  in  ComputerVision  with  widespread  applications.  In  the  field  of  biology,segmentation  has  already  been  used  to  segment  images  ofentire  cells  as  well  as  cellular  sub-components  (e.g.  nuclei).[1], [2],  The quality of a segmentation algorithm, however, ishighly  dependant  on  the  quality  of  the  images  in  the  dataset[site],  preventing  most  neural-network  based  segmentationmethods from identifying and segmenting objects better thanthe  human  eye.  In  other  words,  if  a  human  cannot  detect  anobject  in  an  image,  it’s  unlikely  that  any  object-detection  orsegmentation  algorithm  will  be  able  to  either.  Images  that contain a large amount of noise may mask objects, preventing the  object  from  being  detected  or  segmented.  In  the  case  of segmenting  images  of  nuclei,  there  may  be  a  high  amountof  auto-fluorescence  from  the  microscope  used  to  gather  thei mage,  creating  noise  that  can  mask  cells,  preventing  them from being detected. For  this  project,  I  will  investigate  the  potential  use  of spatial  Raman  spectroscopy  in  segmenting  images  of  nuclei. 




\section{Results}

<img src="images/roc_curves.png">


One main challenge in distinguishing foreground from foreground comes from a large imbalance between the two classes -- there are far greater cell labels than background labels. This is true because the "ground truth" cell labels were generated using a state-of-the art image segmentation method. However, the background spectra were entirely created by hand by one of my collaborators. Due to the labor intensity of generating more background spectra by hand, I explored using Down-sampling and Up-sampling as methods to balance the two classes prior to classifying raman spectra as foreground or background. See \ref{downsample} and \ref{upsample} in the Method Section for more details on the implementations of each. \\

Figure \ref{f1_score} shows accuracies and f1-scores of three different classification methods using two different times of class balancing -- either Up-sampling or Down-sampling. Each of the classifiers perform very poorly on the dataset balanced through up-sampling, performing at or slightly better than a classifier that assigns labels randomly. In stark contrast, the classifiers trained on the dataset balanced through down-sampling perform significantly better, all with accuracies above 70\%. This is further validated my observing the ROC curves in Figure \ref{roc_curves} \\

It's not entirely clear how up-sampling the dataset leads to such poor performance. Further investigation is required to fully understand this. It's also very likely that labeling more background spectra would further improve performance in both cases. \\


\begin{figure*}[t]
\includegraphics[width=15cm]{mapped_img_up.png}
\centering
\caption{
Foreground-background segmentation predictions of 4 random spatial Raman tensors using UPSAMPLED data. Each box shows three columns - the first is the foreground-background mask predicted using a given classifier, the second is a ground truth image (i.e. not Raman), and the third is a segmentation of the ground-truth image using the state-of-the-art nuclei-segmentation algorithm NucleAIzer.  Each row is the prediction using one of three classifiers: 1) Logistic Regression, 2) Random Forests and 3) Fully-Connected 3-Layer Neural Network. E.g. B2) is the prediction of image B using classifier 2, Random Forests, etc.}
\label{mapped_img_up}
\end{figure*}

\subsection{None of the three basic models can segment the test spatial raman spectra near the same level as a state-of-the art model}

Following class-balancing, three basic methods were used to predict whether a Raman spectra in a given spatial pixel belonged to a nuclei or background. Logistic Regression, Random Forests, and a basic shallow Neural Net were used to classify spectra. Figure \ref{f1_score} and \ref{roc_curves} show that Logistic Regression largely out-performed Random Forests and Neural Networks, suggesting that the difference between foreground and background may be captured linearly. \\

The superiority of Logistic Regression also appears in Figure \ref{mapped_img_up}. Of the three classifiers, only logistic regression was capable of predicting any signal when the dataset is balanced through up-sampling -- the other two simply classified the entire image as belonging to the same class.  If the dataset is balanced through down-sampling, however, linear regression seems to perform on par with random forests, both out-performing a shallow neural net, as shown in Figure \ref{mapped_img_down}.  Both Logistic Regression and Random Forest Classifiers are able to capture some large background regions (see Figure \ref{mapped_img_down} A, but all three seem to largely over-classify regions as belonging to nuclei. \\

Despite the dominance of Logistic Regression, none of the three models were able to segment any image with the same quality as the state-of-the-art model, as shown in the third and sixth columns of Figures \ref{mapped_img_up} and \ref{mapped_img_down}. Further thinking was needed to improve performance. \\



\begin{figure*}[t]
\includegraphics[width=15cm]{mapped_img_down.png}
\centering
\caption{Foreground-background segmentation predictions of 4 random spatial Raman tensors using DOWN-SAMPLED data}
\label{mapped_img_down}
\end{figure*}


\subsection{CNN based on tensor patches is superior to basic Models but does not rival state-of-the-art method} 

\begin{figure}
\centering
\includegraphics[width=7cm]{patch_cnn_roc.png}
\caption{ROC curve after training on spectra from 117 images and testing on spectra from 43 images using the image-patch CNN. All spectra were labeled as either foreground or background by hand or by cross-referencing with labels generated from the state-of-the-art segmentation model}
\label{roc_patch_cnn}
\end{figure}

A Convolutional Neural Network that takes into account each Raman spectra's neighbors (see \ref{patch_cnn} for details) was then used. This model far out-performed either of the three basic models, yielding the ROC curve in Figure \ref{roc_patch_cnn} and a test accuracy of 97.6 \%. \\

The trained model was also used to generate segmentations of entire images that were not found in either the train or validation sets.  Figures \ref{mapped_img_patch_cnn} shows the performance of this model on 10 randomly-selected images, as compared with an image taken of the same spatial location and the state-of-the-art segmentation method. This model much more effectively captured nuclei and background than three basic models and even rivals the state-of-the-art segmentation method on some of the images.\\



\begin{figure*}[h]
\centering
\includegraphics[width=\textwidth]{mapped_img_patch_cnn.png}
\caption{Predicted segmentations of 10 randomly-selected images from the test set. The first and fourth columns show the predicted segmentations using a CNN that takes 
Raman tensor patches as input. The second and fifth columns show images taken of the same region and serves as ground-truth. The third and sixth columns show the state-of-the-art segmentations based on these images (i.e. not Raman tensors)}
\label{mapped_img_patch_cnn}
\end{figure*}[h]


\section{Discussion and Summary}
In this project, I have shown that down-sampling the majority class (foreground) prior to classification leads to better performance than up-sampling the minority class (background). I compare several standard classifiers in differentiating foreground from background, showing that Logistic Regression performs better than Random Forests and a Dense 3-Layer Neural Networks in classifying foreground and predicting segmentation masks. I create and show the use of an image patch-based CNN in segmenting, providing great evidence of its superiority in classifying foreground from background and in generating sensible segmentations.  \\

While this model can surely be improved in predicting segmentation masks, this work shows a promising result in segmenting images using only spatial raman tensors instead of images, which may be more informative and better able to capture information than standard nuclei-stained images.  With more time, and perhaps in a future work, I would spend more time hand-labeling background spectra to better balance the foreground and background classes.  This would presumably lead to performance even better than using up-sampling or down-sampling.  It would also provide more data, which nearly always improves a neural net's ability to perform prediction tasks. \\

For future work, I would suggest iterating on the patch-based CNN to better segment images. It may also be desirable to know which features are most important for classifying spectra as foreground or background; applying an interpret-ability method such as Integrated Gradients, Saliency Maps, or Sufficient Input Subsets could help in a deep learning context like this. Lastly, I would propose exploring unsupervised methods for cell-segmentation using spatial Raman tensors in the event labels cannot be effectively acquired.\\

\section{Methods }

\subsection{Dataset}

The dataset used in the project was generated in MIT's Raman Microscopy lab. Over the course of seven days, mouse fibroblast cells undergoing cellular reprogramming were imaged in multiple ways. This project was most concerned with spatial Raman spectral imaging in which a Raman spectra of dimensions 1340 was taken from each pixel in images of dimensions 100 x 100.   Raman tensors were then created of dimension (100, 100, 1,340), one for each time point (taken ever 12 hours). \\

Standard microscopic images were also taken of the cells after their nuclei had been stained for easier visualization. To get the ground-truth labels, we used a nuclei image-segmentation algorithm named NucleAIzer \cite{hollandi_nucleaizer_2020} developed by a colleague at the Broad Institute to get the foreground labels. Another collaborator hand-segmented a portion of the images to get background labels. Only Raman spectra corresponding to pixels that were confidently classified as foreground using NucleAIzer or background by hand-selection were used to create the training, validation, and testing sets. \\

\subsection{Data Preparation}

\subsubsection{Data Preprocessing}
All spectra were pre-processed by first extracting the "fingerprint region" (indices 410:1340 of each spectrum), removing cosmic rays using a recursive version of the Whitaker Haye's algorithm \cite{whitaker_simple_2018} that I developed, removing auto-fluorescence using Rampy's alternative-least squares method \cite{charles_le_losq_rampy_2018}, subtracting a horizontal mean and standard scaling to zero mean and unit variance for each feature (a quantity known as the wavenumber). \\

\subsubsection{Class Balancing \label{upsample} \label{downsample}}
Upsampling was done by sampling background spectra, the minority class, with replacement until the number of background spectra matched the number of foreground spectra in the training, testing, and validation sets. Downsampling was done by random selecting foreground spectra, the majority class, without replacement until the classes were balanced. Sklearn's resampling function was used for both of these class balancing methods.\\


\subsection{Classification}
\subsubsection{Basic Classification Models}
Three basic models were first used to classify Raman spectra from individual pixels as either nuclei (foreground) or background: Logistic Regression, Random Forests, and a shallow Neural Net with three fully-connected hidden layers, each followed by ReLu activation and Dropout layers. The output layer was also fully-connected with two activation units. \\

Logistic Regression and Random Forests were implemented in Python's sklearn using default parameters. The Neural Net was created in Tensorflow and Keras. Optimal Hyperparameters were chosen using Random Search\\


\subsubsection{Patch-CNN}\label{patch_cnn}
I created Patch-CNN, a shallow networks consisting of the following layers: a Convolution Layer with kernel size 2 and 64 activation units, a Max Pooling Layer, two fully-connected hidden layer -- each followed by a Dropout Layer -- and one fully-connected output layer.   Hyperparameters including learning rate, dropout rate, L2 regularization constant,  number of activation units and activation functions were chosen using an optimized version of Random Search, implemented using keras-tuner. \\

Prior to training Patch-CNN, I created the input using Raman tensor patches.  Rather than using the Raman spectra from one pixel alone, I created small tensors of size (3, 3, raman-depth) consisting of a Raman spectra from one pixel and its eight surrounding pixels. If there weren't eight pixels surrounding a given Raman spectra, I would impute any missing spectra with the mean of available surrounding spectra. Each Raman tensor patch had at least four available spectra to use for imputing while a large majority had nine spectra. \\

\subsubsection{Training and Testing}
The training set was made by randomly sampling 60\% of the images. The remaining 40\% of images were divided equally into validation and test sets. The same training, validation, and test sets were used for training and evaluating all four models used in this project. \\

The neural networks were trained using each training set to maximize validation accuracy and area under the ROC curve using the validation set. Adam was used as the optimizer and Binary Cross Entropy as the loss function in both the neural networks\\


\subsubsection{Model Evaluation}
I evaluated each model on the test set consisting of portions of images that had reliable labels. I generated an ROC curve for each model's performance on this smaller test set.  Additionally, I predicted labels (foreground or background) for every pixel in randomly-chosen test images and displayed them next to an image taken of the same spatial location (nuclei stain image) and a state-of-the-art segmentation from NucleAIzer. These representations are shown in Figures \ref{mapped_img_up}, \ref{mapped_img_down}, and \ref{mapped_img_patch_cnn}. \\

\subsubsection{Mapping Predictions to an Image}
Because the outputs of the network were taken from softmax activation, I generated a probability that each spectra belonged to foreground and background. For each spectra (i.e. pixel), I took the probability that it belonged to the foreground and generated an image using these probabilities. The inputs to the model were ordered spatially, making it only necessary to reshape the output probabilities to generate an image. \\
