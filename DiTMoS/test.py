import torch
import numpy as np
import torch.nn.functional as F

def test_classifier(dataloader, classifier, selector, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    classifier.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                selector_pred, selector_feature = selector(X)
            pred= classifier(X, selector_feature)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     print(correct)
    test_loss /= num_batches
    correct /= size
    accuracy = 100*correct
    print(f"Classifier test Error: \n Accuracy: {(accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct

# test union accuracy of the classifiers
def test_union_accuracy(dataloader, classifiers, selector, device):
    correct_number = 0
    total_number = 0
    with torch.no_grad():
        for X, target in dataloader:
            X, target = X.to(device), target.to(device)
            pred = [[] for j in range(len(classifiers))]
            idx = [[] for j in range(len(classifiers))]
            with torch.no_grad():
                selector_pred, selector_feature = selector(X)
            for i in range(len(classifiers)):
                pred[i] = classifiers[i](X, selector_feature)
                pred[i] = pred[i].argmax(1)
                idx[i] = np.squeeze(pred[i].eq(target.data.view_as(pred[i]))).type(torch.float)
            # print(idx)
            for i in range(len(target)):
                if target[i] in [pred[j][i] for j in range(len(classifiers))]:
                    correct_number += 1
                    total_number +=1
                else:
                    total_number +=1
    print(correct_number,total_number)
    print(100*correct_number/total_number)
    return 100*correct_number/total_number

#test the overlap of classifiers
def test_intersection(dataloader, classifiers,selector, device):
    size = len(dataloader.dataset)
    num_classifiers = len(classifiers)
    sample_correct_all = 0
    class_detail = torch.zeros(num_classifiers+1).to(device)
    with torch.no_grad():
        for X, target in dataloader:
            X, target = X.to(device), target.to(device)
            selector_pred, selector_feature = selector(X)
            pred = [[] for j in range(len(classifiers))]
            idx = [[] for j in range(len(classifiers))]
            for i in range(len(classifiers)):
                pred[i] = classifiers[i](X, selector_feature)
                pred[i] = pred[i].argmax(1)
                idx[i] = np.squeeze(pred[i].eq(target.data.view_as(pred[i]))).type(torch.int)
            correct_output = torch.stack(idx, dim=1)
            correct_output = correct_output.sum(dim=1)
            for i in range(len(target)):
                class_detail[correct_output[i]] += 1
            sample_correct_all += correct_output.sum()
    intersection =  sample_correct_all /   (size * num_classifiers)
    print(f" number of correct samples: {num_classifiers}, all samples: {size * num_classifiers}")
    print(f" intersection details: {class_detail}")
    print(100*intersection)
    return intersection,class_detail

def test_ditmos(dataloader, selector, classifiers, device):
    size = len(dataloader.dataset)
    correct = 0
    selector.eval()
    classifiers.eval()
    for batch, (X, y) in enumerate(dataloader):
        X, y= X.to(device), y.to(device)
        pred, feature_map = selector(X)
        # 计算selector分类的train accuracy
        classifiers_pred = []
        classifiers_index = []
        for i in range(len(classifiers)):
            classifier_pred = classifiers[i](X, feature_map)
            classifier_pred = F.softmax(classifier_pred, dim = 1)
            value, pred_index = torch.max(classifier_pred,dim = 1)    
            classifiers_pred.append(value)
            classifiers_index.append(pred_index)
        classifiers_pred = torch.stack(classifiers_pred, dim=1) 
        classifiers_index = torch.stack(classifiers_index, dim=1)
        chosen_index = pred.argmax(1) 
        final_prediction = torch.gather(classifiers_index, 1,chosen_index.reshape(-1,1)) 
        final_prediction = torch.squeeze(final_prediction)
        correct += (final_prediction == y).type(torch.float).sum().item()
    correct /= size
    accuracy = 100 * correct
    print(f"Test DiTMoS Accuracy: {(accuracy):>0.1f}%")
    return correct