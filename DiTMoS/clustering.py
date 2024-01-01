import torch
from torch.utils.data import Dataset
def feature_dataset_for_clustering(dataset, large_model, device = 'cpu'):
    large_model.eval()
    feature = []
    label = []
    large_model = large_model.to(device)
    orig_data = []
    with torch.no_grad():
        for i, (X, y) in enumerate(dataset):
            orig_data.append(X)
            X = X.to(device)
            feature_map = large_model.features(X.reshape(1,3,151)).view(1, -1) 
            feature.append(feature_map.detach().cpu())
            label.append(y)
    return orig_data, feature, label

class cluster_dataset_for_classifier(Dataset):
    def __init__(self, classifier_index,  dataset , class_label, cluster_label, device):
        size = len(dataset)
        self.data_list = []
        self.label_list = []
        # self.transform = transform
        self.device = device
        self.cluster_label = []
        
        for i in range(len(dataset)):
            if cluster_label[i] == classifier_index:
                self.data_list.append(dataset[i])
                self.label_list.append(class_label[i])

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        label = self.label_list[index]
        # data = self.transform(data)
        return data, label