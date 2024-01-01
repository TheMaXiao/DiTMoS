import warnings
import numpy as np
import torch.utils.data as data
warnings.filterwarnings("ignore")
import scipy.io as scio
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

#load the data and labels from the matlab files. 
data = scio.loadmat('datasets/UniMiB-SHAR/data/acc_data.mat')['acc_data']
label = scio.loadmat('datasets/UniMiB-SHAR/data/acc_labels.mat')['acc_labels'][:, 0]   #标签有3轴，第1轴是动作种类

#transform the data to (3, 151) format. 3 represent the x,y,z axis.
data = data.reshape(data.shape[0], 3, 151) 
# print(data.shape)
#there are 17 categories in the UniMiB-SHAR dataset
categories = len(list(Counter(label).keys()))
print(data.shape)

#splitting the dataset into training dataset and testing dataset as 80%:20%.
x_train, x_test, y_train, y_test = [], [], [], []
for i in range(1, categories+1):
    cur_data = data[label == i]
    cur_label = label[label == i]
    cur_x_train, cur_x_test, cur_y_train, cur_y_test = train_test_split(cur_data, cur_label, test_size=0.2, shuffle=True) #split the dataset into training and test sets at 8:2.
    x_train += cur_x_train.tolist()
    x_test += cur_x_test.tolist()
    y_train += cur_y_train.tolist()
    y_test += cur_y_test.tolist()

#Save the training and testing dataset as npy files.
np.save('datasets/UniMiB-SHAR/x_train', np.array(x_train))
np.save('datasets/UniMiB-SHAR/x_test', np.array(x_test))
np.save('datasets/UniMiB-SHAR/y_train', np.array(y_train)-1)
np.save('datasets/UniMiB-SHAR/y_test', np.array(y_test)-1)