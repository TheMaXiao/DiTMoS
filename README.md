### [DiTMoS: Delving into Diverse Tiny-Model Selection on Microcontrollers]

## Description

This is an implementation example of DiTMoS(server side). DiTMoS is a framework to utilize a set of tiny models to boost the accuracy on time-series mobile applications under comparable memory and latency constraints. 

**Our paper has been submitted to [PerCom'2024](https://www.percom.org/)** .

## Requirements

The DiTMoS code is realized by PyTorch 3.10.11.
You need to install SciPy, scikit-learn library.

## How to Run DiTMoS

The code is an full example of DiTMoS implementation on UniMiB-SHAR dataset which has 17 classes for human activity recognition. The dataset can be found in datasets folder. SInce the UniMiB-SHAR dataset is collected by Matlab, we provide a pre-processing module to convert the Matlab version to python version and split the full dataset to training and test sets as 80%:20%.

1. Run UniMiB-preprocessing.py from pre-processing folder to create training and testing datasets.

Then we can 