import os
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def init_weights(net_layer):
    try:
        net_layer.apply(weights_init_normal)
    except:
        raise NotImplementedError('weights initialization error')

class numericalRegression(nn.Module):
    def __init__(self, h0 = 24, c0 = 2):
        super(numericalRegression, self).__init__()

        self.h0 = h0
        self.c0 = c0

        self.gru = nn.GRU(input_size = 1, hidden_size = self.h0, num_layers = self.c0, batch_first = True)
        if isinstance(self.gru, nn.GRU):
            for param in self.gru.parameters():
                nn.init.normal_(param.data)

        self.linear = nn.Linear(self.h0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        '''if torch.cuda.is_available():
            self.h0 = torch.zeros(self.c0, x.size(0), self.h0).cuda()
        else:
            self.h0 = torch.zeros(self.c0, x.size(0), self.h0)#x.size(0) = args.batch_size

        if torch.cuda.is_available():
            self.c0 = torch.zeros(self.c0, x.size(0), self.h0).cuda()
        else:
            self.c0 = torch.zeros(self.c0, x.size(0), self.h0)'''
        
        out,hn = self.gru(x.view(x.size(0) ,-1 ,1))
        x = self.linear(hn[-1])
        x = self.sigmoid(x)
        return x



class textualRegression(nn.Module):
    def __init__(self):
        super(textualRegression, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 1, out_channels = 4, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.Dropout(0.5))
        init_weights(self.conv1)
        
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = 4, out_channels = 4, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 4, out_channels = 8, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.Dropout(0.5))
        init_weights(self.conv2)
        
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = 8, out_channels = 8, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.Dropout(0.5))
        init_weights(self.conv3)
        
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.Dropout(0.5))
        init_weights(self.conv4)
        
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
                                         nn.ReLU(True),
                                         nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
                                         nn.ReLU(True),
                                         nn.Dropout(0.5))
        init_weights(self.conv5)

        self.linear = nn.Sequential(nn.Linear(128 * 1 * 32, 1),
                                    nn.ReLU(True),
                                    nn.Dropout(0.5))

        init_weights(self.linear)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(-1, 128 * 1 * 32)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
        

class mixRegression(nn.Module):
    def __init__(self):
        super(mixRegression, self).__init__()
        self.linear = nn.Linear(2, 1)
        init_weights(self.linear)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
