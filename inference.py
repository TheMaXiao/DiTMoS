from model import NeuralNetwork, Tiny_NeuralNetwork_Classifier, Tiny_NeuralNetwork_Selector
from DiTMoS.clustering import feature_dataset_for_clustering, cluster_dataset_for_classifier
from DiTMoS.adversarial_training import selector_train, classifier_pretrain, DiTMoS_training_framework, train_classifier_from_selector
from DiTMoS.test import test_classifier, test_ditmos, test_union_accuracy
import torch
from torch import nn
import warnings
import numpy as np
import os
from pathlib import Path
import torch.utils.data as data
warnings.filterwarnings("ignore")
import scipy.io as scio
from collections import Counter
import os

num_classifiers = 6
test_batch_size = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_x = torch.from_numpy(np.load('datasets/UniMiB-SHAR/x_train.npy')).float()
train_y = torch.from_numpy(np.load('datasets/UniMiB-SHAR/y_train.npy')).long()
test_x  = torch.from_numpy(np.load('datasets/UniMiB-SHAR/x_test.npy')).float()
test_y  = torch.from_numpy(np.load('datasets/UniMiB-SHAR/y_test.npy')).long()
num_classes = len(Counter(train_y.tolist()))
len_train, len_test = len(train_y),  len(test_y)
print(num_classes,len_train, len_test,train_x.shape, train_y.shape,test_x.shape,test_y.shape )

train_set = data.TensorDataset(train_x, train_y)
test_set = data.TensorDataset(test_x, test_y)
test_loader_adv = data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)
# load models from saved_model folder.
classifiers = nn.ModuleList([Tiny_NeuralNetwork_Classifier(output_size=num_classes, conv = [8,8,4], fc = 64) for i in range(num_classifiers)]).to(device)
selector = Tiny_NeuralNetwork_Selector(output_size=num_classifiers, conv = [8,8,4], fc = 64).to(device)
classifiers = torch.load("./saved_model/ditmos_classifiers_nclassifier_6.pth")
selector = torch.load("./saved_model/ditmos_selector_nclassifier_6.pth")

#test DiTMoS
test_correct = test_ditmos(test_loader_adv, selector, classifiers,device)

print(f'The accuracy of DiTMoS on UniMiB dataset is {100*test_correct:>0.1f}%.')