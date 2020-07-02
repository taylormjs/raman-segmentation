---
layout: default
title: "dataset"
---

### Dataset

The dataset used in the project was generated in MIT's Raman Microscopy lab. Over the course of seven days, mouse fibroblast cells undergoing cellular reprogramming were imaged in multiple ways. This project was most concerned with spatial Raman spectral imaging in which a Raman spectra of dimensions 1340 was taken from each pixel in images of dimensions 100 x 100.   Raman tensors were then created of dimension (100, 100, 1,340), one for each time point (taken ever 12 hours). \\

Standard microscopic images were also taken of the cells after their nuclei had been stained for easier visualization. To get the ground-truth labels, we used a nuclei image-segmentation algorithm named NucleAIzer \cite{hollandi_nucleaizer_2020} developed by a colleague at the Broad Institute to get the foreground labels. Another collaborator hand-segmented a portion of the images to get background labels. Only Raman spectra corresponding to pixels that were confidently classified as foreground using NucleAIzer or background by hand-selection were used to create the training, validation, and testing sets. \\


B.  Data Preparation1)  Data  Preprocessing:All  spectra  were  pre-processedby  first  extracting  the  ”fingerprint  region”  (indices  410:1340of  each  spectrum),  removing  cosmic  rays  using  a  recursiveversion of the Whitaker Haye’s algorithm [7] that I developed,removing  auto-fluorescence  using  Rampy’s  alternative-leastsquares   method   [8],   subtracting   a   horizontal   mean   andstandard  scaling  to  zero  mean  and  unit  variance  for  eachfeature (a quantity known as the wavenumber).2)  Class  Balancing  :Upsampling  was  done  by  samplingbackground spectra, the minority class, with replacement untilthe  number  of  background  spectra  matched  the  number  offoreground spectra in the training, testing, and validation sets.Downsampling  was  done  by  random  selecting  foregroundspectra,  the  majority  class,  without  replacement  until  theclasses  were  balanced.  Sklearn’s  resampling  function  wasused for both of these class balancing methods
