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
import random
from sklearn.cluster import KMeans

#set the hyperparameters number of classifiers
num_classifiers = 6

strong_model_batch_size = 8 #pretrain strong model batch size
adversarial_batch_size = 8 #adversarial training batch size

pretrain_classifiers_epochs = 30 #pre-train classifier on subsets
pretrain_selector_epochs = 20 #pre-train selector from pretrained classifiers

adversarial_iterations = 40 #The number of adversarial training iterations. Each iteration consists of a classifier training step and a selector training step. 

selector_step_epoch = 6 #the number of epochs in the selector training step.
classifier_step_epoch = 6 #the number of epochs in the classifier training step.

#In the loss parameters, there are 4 factors in the equation 2 in the paper.
#1. Union loss, 2. Overlap loss, 3. Single loss 4. Selection loss.
loss_parameter = [0.1,0.03,0.1,1]

#activate cuda or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#set seed for reproduce
seed = 1
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#load dataset and create dataloader

train_x = torch.from_numpy(np.load('datasets/UniMiB-SHAR/x_train.npy')).float()
train_y = torch.from_numpy(np.load('datasets/UniMiB-SHAR/y_train.npy')).long()
test_x  = torch.from_numpy(np.load('datasets/UniMiB-SHAR/x_test.npy')).float()
test_y  = torch.from_numpy(np.load('datasets/UniMiB-SHAR/y_test.npy')).long()
num_classes = len(Counter(train_y.tolist()))
len_train, len_test = len(train_y),  len(test_y)
print(num_classes,len_train, len_test,train_x.shape, train_y.shape,test_x.shape,test_y.shape )

train_set = data.TensorDataset(train_x, train_y)
test_set = data.TensorDataset(test_x, test_y)
train_loader = data.DataLoader(dataset=train_set, batch_size=strong_model_batch_size, shuffle=True)
test_loader = data.DataLoader(dataset=test_set, batch_size=strong_model_batch_size, shuffle=True)
train_loader_adv = data.DataLoader(dataset=train_set, batch_size=adversarial_batch_size, shuffle=True)
test_loader_adv = data.DataLoader(dataset=test_set, batch_size=adversarial_batch_size, shuffle=True)
#Create Strong model and pre-train model on the dataset
large_model = NeuralNetwork(output_size=num_classes, conv = [64,64,32,32,32,16]).to(device)


file_path = "./pretrained_model/pretrain_strong_model_UniMiB.pth"

# If there is a pretrained strong model, then load it from the folder. If not, train a new pretrain model and save it to the folder

if os.path.exists(file_path):
    print('Pre-trained model exists!')
    large_model = torch.load("./pretrained_model/pretrain_strong_model_UniMiB.pth") 
    correct = 0
    test_loss = 0
    loss_fn_pretrain = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = large_model(X)
            test_loss += loss_fn_pretrain(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= len(test_loader)
    correct /= len_test
    accuracy = 100*correct
    print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
else:
    print('No pre-trained model exists! Now train a strong model!')
    #train the strong model from scratch.
    # pretrain_optimizer = torch.optim.SGD(large_model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(pretrain_optimizer, 20, gamma=0.2, last_epoch=-1)
    loss_fn_pretrain = nn.CrossEntropyLoss()
    epochs = 50
    best_test_acc = 0
    for epoch in range(epochs):
        print(f'Strong model training: epoch {epoch+1} of {epochs}')
        train_loss = 0
        correct = 0
        test_loss = 0
        learning_rate = 0.005 * (0.2 ** (epoch // 20))
        pretrain_optimizer = torch.optim.SGD(large_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            #computing prediction error
            pred= large_model(X)
            loss = loss_fn_pretrain(pred, y)
        #         print(loss, loss.item())
            train_loss += loss
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            #backprop
            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()
        correct /= len_train
        accuracy = 100*correct
        print(f"Train Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                pred = large_model(X)
                test_loss += loss_fn_pretrain(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= len(test_loader)
        correct /= len_test
        accuracy = 100*correct
        print(f"Test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
        if accuracy >= best_test_acc:
            best_test_acc = accuracy
            torch.save(large_model, './pretrained_model/pretrain_strong_model_UniMiB.pth')
        print(f'The best test accuracy of the strong model is {(best_test_acc):>0.1f}%.')
    
#Feature extraction from large strong model
orig_data, large_model_feature, large_model_label = feature_dataset_for_clustering(train_set, large_model)
train_feature_map = torch.cat(large_model_feature, dim=0)
train_feature_map = train_feature_map.numpy()
test_orig_data, test_large_model_feature, test_large_model_label = feature_dataset_for_clustering(test_set, large_model)
test_feature_map = torch.cat(test_large_model_feature, dim=0)
test_feature_map = test_feature_map.numpy()


kmeans_classifer = KMeans(n_clusters=num_classifiers, random_state=0, n_init=50)
kmeans_classifer.fit(train_feature_map)

train_cluster = kmeans_classifer.predict(train_feature_map)
test_cluster = kmeans_classifer.predict(test_feature_map)

#train_classifiers
train_classifier_set = []
test_classifier_set = []
train_classifier_loader = []
test_classifier_loader = []
for i in range(num_classifiers):
    train_classifier_set.append(cluster_dataset_for_classifier(classifier_index = i, dataset = orig_data, class_label = large_model_label, cluster_label = train_cluster, device = device))
    test_classifier_set.append(cluster_dataset_for_classifier(classifier_index = i,  dataset = test_orig_data, class_label = test_large_model_label, cluster_label = test_cluster, device = device))
    train_classifier_loader.append(torch.utils.data.DataLoader(train_classifier_set[i], batch_size = 8, shuffle=True))
    test_classifier_loader.append(torch.utils.data.DataLoader(test_classifier_set[i], batch_size = 8, shuffle=False))

# define classifiers and selector
classifiers = nn.ModuleList([Tiny_NeuralNetwork_Classifier(output_size=num_classes, conv = [8,8,4], fc = 64) for i in range(num_classifiers)]).to(device)
selector = Tiny_NeuralNetwork_Selector(output_size=num_classifiers, conv = [8,8,4], fc = 64).to(device)

# classifiers pre-train
loss_fn = nn.CrossEntropyLoss()


for i in range(len(classifiers)):
    for t in range(pretrain_classifiers_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        learning_rate = 0.01 * (0.5 ** (t // 6))
        optimizer = torch.optim.SGD(classifiers[i].parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        loss, train_correct = classifier_pretrain(train_classifier_loader[i], classifiers[i], selector, loss_fn, optimizer,device)
        test_correct = test_classifier(test_classifier_loader[i], classifiers[i], selector, loss_fn, device) 



# Selector pretrain
for t in range(pretrain_selector_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    selector_learning_rate = 0.01 * (0.5 ** (t // 8))
    selector_optimizer = torch.optim.SGD(selector.parameters(), lr=selector_learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
    train_ce_loss, train_correct = selector_train(train_loader, selector, classifiers, selector_optimizer,device)
    test_correct = test_ditmos(test_loader, selector, classifiers, device)

#define DiTMoS framework
model = DiTMoS_training_framework(classifiers,selector,device).to(device)

# adversarial training:
# The adversarial training phase consists of selector training step and classifier training step. Each step contains several epochs.
test_acc_list = [0]
classifier_optimizer = []
for i in range(num_classifiers):
    classifier_optimizer.append(torch.optim.SGD(classifiers[i].parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4, nesterov=True))
for p in range(adversarial_iterations):
    print(f"Adversarial iteration {p+1} start\n-------------------------------")
    for t in range(selector_step_epoch):
    #gate_train
        print(f"iteration: {p+1}, gate epoch: {t+1}")
        selector_learning_rate = 0.001 * (0.5 ** ( (selector_step_epoch *p+t) // 30))
        selector_optimizer = torch.optim.SGD(selector.parameters(), lr=selector_learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        train_ce_loss, train_correct = selector_train(train_loader, selector, classifiers, selector_optimizer,device)
        test_correct = test_ditmos(test_loader, selector, classifiers,device)
        if test_correct >= max(test_acc_list):
            torch.save(selector, './saved_model/ditmos_selector_nclassifier_'+str(num_classifiers)+'.pth')
            torch.save(classifiers, './saved_model/ditmos_classifiers_nclassifier_'+str(num_classifiers)+'.pth')
        test_acc_list.append(test_correct)
    
    for t in range(classifier_step_epoch):
        classifier_learning_rate = 0.001 * (0.5 ** ((classifier_step_epoch*p+t) // 30))
        print(f"classifier training iteration {p+1} epoch {t+1}: ")
        for i in range(num_classifiers):
            classifier_optimizer[i] = torch.optim.SGD(classifiers[i].parameters(), lr=classifier_learning_rate, momentum=0.9, weight_decay=1e-4, nesterov=True)
        train_weight_loss, train_correct = train_classifier_from_selector(train_loader_adv, selector, model, classifier_optimizer, loss_weights = loss_parameter,device = device)
        
        test_correct = test_ditmos(test_loader_adv, selector, classifiers,device)
        print(f"iteration :{p+1}, classifier epoch: {t+1} done!")
        if test_correct >= max(test_acc_list):
            torch.save(selector, './saved_model/ditmos_selector_nclassifier_'+str(num_classifiers)+'.pth')
            torch.save(classifiers, './saved_model/ditmos_classifiers_nclassifier_'+str(num_classifiers)+'.pth')
        test_acc_list.append(test_correct)
    test_union_accuracy(test_loader, classifiers, selector, device)

print(f'The accuracy of DiTMoS on UniMiB dataset is {100*max(test_acc_list):>0.1f}%.')
