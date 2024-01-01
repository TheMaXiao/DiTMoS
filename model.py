import torch
from torch import nn

#Define Strong model
class NeuralNetwork(nn.Module):
    def __init__(self, output_size=17, conv = [64,64,32,32,32,16], fc = 128):
        super(NeuralNetwork, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=3)
        self.features = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=conv[0], kernel_size=15, padding = 'same'),
            nn.BatchNorm1d(conv[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv[0], out_channels=conv[1], kernel_size=13, padding = 'same'),
            nn.BatchNorm1d(conv[1]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            nn.Conv1d(in_channels=conv[1], out_channels=conv[2], kernel_size=9, padding = 'same'),
            nn.BatchNorm1d(conv[2]),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv[2], out_channels=conv[3], kernel_size=9, padding = 'same'),
            nn.BatchNorm1d(conv[3]),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=conv[3], out_channels=conv[4], kernel_size=5, padding = 'same'),
            nn.BatchNorm1d(conv[4]),
            nn.ReLU(),            
            nn.Conv1d(in_channels=conv[4], out_channels=conv[5], kernel_size=5, padding = 'same'),
            nn.BatchNorm1d(conv[5]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )  
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=conv[5]*18, out_features=fc),
            nn.ReLU())        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc, out_features=output_size))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)  
        out = self.dropout(out)   
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
#Define Strong model    
class Tiny_NeuralNetwork_Selector(nn.Module):
    def __init__(self, output_size=17, conv = [8,8,4], fc = 96):
        super(Tiny_NeuralNetwork_Selector, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=3)
        self.layer_1 = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=conv[0], kernel_size=15),
            nn.BatchNorm1d(conv[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=conv[0], out_channels=conv[1], kernel_size=9),
            nn.BatchNorm1d(conv[1]),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer_3 = nn.Sequential(    
            nn.Conv1d(in_channels=conv[1], out_channels=conv[2], kernel_size=5),
            nn.BatchNorm1d(conv[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )  
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=conv[2]*13, out_features=fc),
            nn.ReLU())        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc, out_features=output_size))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x):
        layer1 = self.layer_1(x)
        layer2 = self.layer_2(layer1)
        layer3 = self.layer_3(layer2)
        out = layer3.reshape(layer3.size(0), -1)  
        out = self.dropout(out) 
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out, [layer1, layer2]
    

class Tiny_NeuralNetwork_Classifier(nn.Module):
    def __init__(self, output_size=17, conv = [16,8,8], fc = 96):
        super(Tiny_NeuralNetwork_Classifier, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=3)
        self.layer_1 = nn.Sequential(
            nn.BatchNorm1d(3),
            nn.Conv1d(in_channels=3, out_channels=conv[0], kernel_size=15),
            nn.BatchNorm1d(conv[0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2))
        
        self.layer_2 = nn.Sequential(
            nn.Conv1d(in_channels=conv[0]*2 , out_channels=conv[1], kernel_size=9),
            nn.BatchNorm1d(conv[1]),
            nn.ReLU(),            
            nn.MaxPool1d(kernel_size=2, stride=2))
        self.layer_3 = nn.Sequential(    
            nn.Conv1d(in_channels=conv[1]*2, out_channels=conv[2], kernel_size=5),
            nn.BatchNorm1d(conv[2]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )  
        self.fc1 = nn.Sequential(
            nn.Linear(in_features=conv[2]*13, out_features=fc),
            nn.ReLU())        
        self.fc2 = nn.Sequential(
            nn.Linear(in_features=fc, out_features=output_size))
        self.dropout = nn.Dropout(0.2)
    def forward(self, x, gate_feature):
        layer1 = self.layer_1(x)
        layer2 = self.layer_2(torch.cat((layer1 , gate_feature[0]), 1))
        layer3 = self.layer_3(torch.cat((layer2 , gate_feature[1]), 1))
        self.feature_map = layer3.detach()
        out = layer3.reshape(layer3.size(0), -1)  
        out = self.dropout(out) 
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out