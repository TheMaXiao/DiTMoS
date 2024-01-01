import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DiTMoS_training_framework(nn.Module):
    def __init__(self, classifiers, selector, device):
        super(DiTMoS_training_framework, self).__init__()
        # instantiate classifiers
        self.template_classifiers = deepcopy(classifiers)
        self.classifiers = classifiers
        self.num_classifiers = len(classifiers)
        self.selector = selector
        self.device = device
    #这个loss的思路： 对于一个样本，统计哪几个样本是可以分对的，然后对于正确的样本，只选择一个进行训练
    def intersection_union_loss(self, classifier_outputs, y, softmax_template):
        pred_value_list = []
        pred_correct_list = []
        ce_loss = nn.CrossEntropyLoss(reduction = "none")
        #需要通过筛选，将需要训练的模型选出来，筛选过程的函数不能带grad
        for i in range(self.num_classifiers):
            classifier_confidence, classifier_pred = torch.max(softmax_template[i], dim=1)
            classifier_correct = (classifier_pred == y).type(torch.float)
            pred_correct_list.append(classifier_correct)
            pred_value_list.append(classifier_confidence)
        #pred_correct_list是有用的，list中的每一个代表了它能不能被分对
        pred_value_list = torch.stack(pred_value_list, dim=1)
        pred_correct_list_to_tensor = torch.stack(pred_correct_list, dim=1)

        correct_num = torch.squeeze(pred_correct_list_to_tensor.sum(dim=1)) #[batch]，每一个数值相当于该样本可以被几个classifiers分对
        mask_wrong = torch.squeeze(torch.where(correct_num == 0, 1, 0))
        mask_multi_correct = torch.squeeze(torch.where(correct_num > 1, 1, 0))
        mask_one_correct = torch.squeeze(torch.where(correct_num == 1, 1, 0))
        num_wrong_batch = mask_wrong.sum().type(torch.float)
        num_multi_batch = mask_multi_correct.sum().type(torch.float)
        num_one_batch = mask_one_correct.sum().type(torch.float)
        #针对三种不同的情况分别设计loss,最终的型式应该是，每一个loss形成一个列表，每个列表是【num_classifiers】结构，里面每个元素是对应classifier的loss
        #1. wrong_loss
        wrong_loss = []
        for i in range(self.num_classifiers):
            batch_wrong_loss = ce_loss(classifier_outputs[i], y)
            num_samples = torch.maximum(num_wrong_batch, torch.tensor(1).to(self.device))
            batch_wrong_loss = (mask_wrong * batch_wrong_loss).sum()
            # print(f"batch_wrong_loss {batch_wrong_loss.shape}")
            batch_wrong_loss = batch_wrong_loss/num_samples
            wrong_loss.append(batch_wrong_loss)
        #2. multi_correct_loss
        multi_correct_loss = []
        for i in range(self.num_classifiers):
            classifier_multi_correct = pred_correct_list[i] * mask_multi_correct
            num_samples = torch.maximum(classifier_multi_correct.sum(), torch.tensor(1).to(self.device))
            batch_multi_loss = ce_loss(classifier_outputs[i], y)
            batch_multi_loss = (classifier_multi_correct * batch_multi_loss).sum()
            batch_multi_loss = batch_multi_loss/num_samples
            multi_correct_loss.append(-1 * batch_multi_loss)
        #3. single correct loss
        one_correct_loss = []
        for i in range(self.num_classifiers):
            classifier_unique_correct = pred_correct_list[i] * mask_one_correct
            num_samples = torch.maximum(classifier_unique_correct.sum(), torch.tensor(1).to(self.device))
            batch_one_loss = ce_loss(classifier_outputs[i], y)
            batch_one_loss = (classifier_unique_correct * batch_one_loss).sum()
            batch_one_loss = batch_one_loss/num_samples
            one_correct_loss.append(batch_one_loss)
        return wrong_loss, multi_correct_loss, one_correct_loss
    
    def forward(self, x, y):
        self.template_classifiers.eval()
        self.selector.eval()
        with torch.no_grad():
            selector_pred, selector_feature = self.selector(x)
            softmax_template = [self.classifiers[i](x, selector_feature) for i in range(self.num_classifiers)]
        classifier_outputs = [self.classifiers[i](x, selector_feature) for i in range(self.num_classifiers)]
        wrong_loss, multi_correct_loss, one_correct_loss = self.intersection_union_loss(classifier_outputs, y, softmax_template)
        return classifier_outputs, wrong_loss, multi_correct_loss, one_correct_loss



def classifier_pretrain(dataloader, model, selector,  loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    correct = 0
    model.train()
    num_batches = len(dataloader)
    train_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #computing prediction error
        with torch.no_grad():
            selector_pred, selector_feature = selector(X)
        pred= model(X, selector_feature)
        loss = loss_fn(pred, y)
#         print(loss, loss.item())
        train_loss += loss
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        #backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
    correct /= size
    train_loss /= num_batches
    accuracy = 100*correct
    print(f"Pretrain Accuracy of Classifier: {(accuracy):>0.1f}%, Avg loss: {train_loss:>0.2f}")      
    return correct, train_loss

def train_classifier_from_selector(dataloader, selector,  DiTMoS_training_framework, optimizer, loss_weights = [1, 1, 1, 1], device = 'cpu'):
    size = len(dataloader.dataset)
    correct = 0
    selector.eval()
    classifiers = DiTMoS_training_framework.classifiers
    num_batches = len(dataloader)
    classifiers.train()
    classifier_ce_loss = nn.CrossEntropyLoss(reduction="none")
    num_classifiers = len(classifiers)
    train_classifier_loss = torch.zeros(num_classifiers).to(device)

    for batch, (X, y) in enumerate(dataloader):
        X, y= X.to(device), y.to(device)
        with torch.no_grad():
            pred, feature_map = selector(X)
        pred = F.softmax(pred, dim=1)
        chosen_index = pred.argmax(1).reshape(X.size(0),1) #指的是选择哪一个classifier
        #计算classifier_loss
        zeros = torch.zeros(X.size(0), num_classifiers).to(device)
        classifier_weight_matrix = zeros.scatter(1, chosen_index, 1)
        classifier_preds, wrong_loss, multi_correct_loss, one_correct_loss = DiTMoS_training_framework(X,y)
        for i in range(num_classifiers):
            classifier_weight = classifier_weight_matrix[:,i]
            # print(classifier_weight)
            classifier_pred = classifiers[i](X, feature_map)
            correct_number = torch.maximum(torch.tensor(1),classifier_weight.sum())
            classifier_loss = classifier_ce_loss(classifier_pred, y)
            weighted_loss = (classifier_loss * classifier_weight).sum()/correct_number
            # weighted_loss = (classifier_loss * classifier_weight).mean()
            loss = loss_weights[0] * wrong_loss[i] + loss_weights[1] * multi_correct_loss[i] + loss_weights[2] * one_correct_loss[i] + loss_weights[3] * weighted_loss
            optimizer[i].zero_grad()
            loss.backward()
            optimizer[i].step()
            train_classifier_loss[i] += weighted_loss
        # 计算selector分类的train accuracy
        classifiers_pred = []
        classifiers_index = []
        for i in range(len(classifiers)):
            classifier_pred = classifiers[i](X, feature_map)
            classifier_pred = F.softmax(classifier_pred, dim = 1)
            value, pred_index = torch.max(classifier_pred,dim = 1)    
            classifiers_pred.append(value)
            classifiers_index.append(pred_index)
        classifiers_pred = torch.stack(classifiers_pred, dim=1) #tensor shape [batch, classifier_num]
        classifiers_index = torch.stack(classifiers_index, dim=1)
        chosen_index = pred.argmax(1) #chosen_index indicates the selected classifier by selector
        final_prediction = torch.gather(classifiers_index, 1,chosen_index.reshape(-1,1)) 
        final_prediction = torch.squeeze(final_prediction)
        correct += (final_prediction == y).type(torch.float).sum().item()
    correct /= size
    train_classifier_loss /= num_batches
    accuracy = 100 * correct
    print(f"Train Accuracy of Classifier Step: {(accuracy):>0.1f}%")
    return train_classifier_loss, correct

def selector_train(dataloader, selector, classifiers, optimizer, device = 'cpu'):
    size = len(dataloader.dataset)
    correct = 0
    selector.train()
    num_batches = len(dataloader)
    num_classifiers = len(classifiers)
    classifiers.eval()
    m = nn.Sigmoid()
    selector_ce_loss = nn.CrossEntropyLoss()
    train_ce_loss = 0
    for batch, (X, y) in enumerate(dataloader):
        X,  y = X.to(device), y.to(device)
        pred, feature_map = selector(X)
        #claculate selector_bce_loss
        classifiers_pred_value = []
        classifiers_index = []
        with torch.no_grad():
            for i in range(num_classifiers):
                classifier_pred = classifiers[i](X, feature_map)
                classifier_pred = F.softmax(classifier_pred, dim = 1)
                value, index = torch.max(classifier_pred,dim = 1)
                binary_label = (index == y).type(torch.LongTensor).to(device)
                classifiers_index.append(index)
                soft_label = torch.where(binary_label == 1, value, -1 * value).to(device)
                classifiers_pred_value.append(soft_label)
            classifiers_index = torch.stack(classifiers_index, dim=1)
            classifiers_pred_value = torch.stack(classifiers_pred_value, dim=1)
            # print(classifiers_pred_value.shape)
            single_label = classifiers_pred_value.argmax(1)
            # print(single_label)
        ce_loss = selector_ce_loss(pred, single_label)
        #计算class label loss
        loss = ce_loss
        train_ce_loss += ce_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算selector分类的train accuracy
        chosen_index = pred.argmax(1) #指的是选择哪一个classifier
        final_prediction = torch.gather(classifiers_index, 1,chosen_index.reshape(-1,1)) #check对不对
        final_prediction = torch.squeeze(final_prediction)
        correct += (final_prediction == y).type(torch.float).sum().item()
        # if batch % 1000 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
   
    correct /= size
    train_ce_loss /= num_batches
    accuracy = 100 * correct
    print(f"Train Accuracy of Selector training: {(accuracy):>0.1f}%, Avg loss: {train_ce_loss:>0.2f}")
    return train_ce_loss, correct