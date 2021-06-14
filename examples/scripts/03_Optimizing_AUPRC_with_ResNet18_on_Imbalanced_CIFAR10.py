"""
Author: Zhuoning Yuan, Qi Qi
Contact: yzhuoning@gmail.com, qi-qi@uiowa.edu
"""
from libauc.losses import SOAPLoss
from libauc.optimizers import SGD
from libauc.models import ResNet18
from libauc.datasets import CIFAR10
from libauc.datasets import ImbalanceGenerator, ImbalanceSampler 

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import numpy as np
import torch
from PIL import Image

"""# **Reproducibility**"""

def set_all_seeds(SEED):
    # REPRODUCIBILITY
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""# **Image Dataset**"""

class ImageDataset(Dataset):
    def __init__(self, images, targets, image_size=32, crop_size=30, mode='train'):
       self.images = images.astype(np.uint8)
       self.targets = targets
       self.mode = mode
       self.transform_train = transforms.Compose([                                                
                              transforms.ToTensor(),
                              transforms.RandomCrop((crop_size, crop_size), padding=None),
                              transforms.RandomHorizontalFlip(),
                              transforms.Resize((image_size, image_size)),
                              
                              ])
       self.transform_test = transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Resize((image_size, image_size)),
                              ])
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]
        image = Image.fromarray(image.astype('uint8'))
        if self.mode == 'train':
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)
        return idx, image, target

"""# **Paramaters**"""

# paramaters
imratio = 0.02
SEED = 123
BATCH_SIZE = 64
lr =  0.6
weight_decay = 2e-4
margin = 0.5
gamma = 0.99
posNum = 1

"""# **Loading datasets**"""

# dataloader 
(train_data, train_label), (test_data, test_label) = CIFAR10()
(train_images, train_labels) = ImbalanceGenerator(train_data, train_label, imratio=imratio, shuffle=True, random_seed=SEED)
(test_images, test_labels) = ImbalanceGenerator(test_data, test_label, is_balanced=True,  random_seed=SEED)

train_dataset = ImageDataset(train_images, train_labels)
test_dataset = ImageDataset(test_images, test_labels, mode='test')
testloader = torch.utils.data.DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False, num_workers=1,  pin_memory=True)

"""# **Creating models & AUC Optimizer**"""

set_all_seeds(456)
model = ResNet18(pretrained=False, last_activation=None)  # last_activation=False: don't include sigmoid function in the last layer
model = model.cuda()

Loss = SOAPLoss(margin=margin, gamma=gamma, data_len=train_labels.shape[0])
optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

"""# **Training**"""

# training 
model.train()
losses = []  
print ('-'*30)
total_iters = 0
for epoch in range(64):
    if epoch == 32:
       optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
    
    train_pred = []
    train_true = []
    model.train() 
       
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=ImbalanceSampler(train_labels.flatten().astype(int), BATCH_SIZE, pos_num=posNum), num_workers=2, pin_memory=True, drop_last=True) 

    for idx, (index, data, targets) in enumerate(trainloader):
        data, targets  = data.cuda(), targets.cuda()
        y_pred = model(data)
        predScore = torch.sigmoid(y_pred)

        loss = Loss(predScore, targets, index_s=index)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_pred.append(predScore.cpu().detach().numpy())
        train_true.append(targets.cpu().detach().numpy())

    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    train_auc = roc_auc_score(train_true, train_pred) 
    train_prc = average_precision_score(train_true, train_pred)

    model.eval()
    test_pred = []
    test_true = [] 
    for j, data in enumerate(testloader):
        _, test_data, test_targets = data
        test_data = test_data.cuda()
        y_pred = model(test_data)
        y_pred = torch.sigmoid(y_pred)
        test_pred.append(y_pred.cpu().detach().numpy())
        test_true.append(test_targets.numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
     
    val_auc =  roc_auc_score(test_true, test_pred) 
    val_prc = average_precision_score(test_true, test_pred)
    model.train()
    print("epoch: {}, train_loss: {:4f}, train_ap:{:4f}, test_ap:{:4f},  lr:{:4f}".format(epoch, loss.item(), train_prc, val_prc,  optimizer.param_groups[0]['lr'] ))