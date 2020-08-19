import torch
import torch.nn as nn


class discriminator(nn.Module):
    def __init__(self,imgChannels,featuresDiscrim):
        super(discriminator,self).__init__()
        self.net=nn.Sequential(
            nn.Conv2d(imgChannels,featuresDiscrim,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(featuresDiscrim,featuresDiscrim*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(featuresDiscrim*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(featuresDiscrim*2,featuresDiscrim*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(featuresDiscrim*4),
            nn.LeakyReLU(0.2),
             nn.Conv2d(featuresDiscrim*4,featuresDiscrim*8,kernel_size=4,stride=2,padding=1),
             nn.BatchNorm2d(featuresDiscrim*8),
             nn.LeakyReLU(),
             nn.Conv2d(featuresDiscrim*8,1,kernel_size=4,stride=2,padding=0),
             nn.Sigmoid()
        )
    def forward(self,x):        
        return self.net(x)



class generator(nn.Module):
    def __init__(self,channelsNoise,channelsImg,featuresGen):
        super(generator,self).__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(channelsNoise,featuresGen*16,kernel_size=4,stride=1,padding=0),
            nn.BatchNorm2d(featuresGen*16),
            nn.ReLU(),

            nn.ConvTranspose2d(featuresGen*16,featuresGen*8,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(featuresGen*8),
            nn.ReLU(),

            nn.ConvTranspose2d(featuresGen*8,featuresGen*4,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(featuresGen*4),
            nn.ReLU(),

            nn.ConvTranspose2d(featuresGen*4,featuresGen*2,kernel_size=4,stride=2,padding=1),
            nn.BatchNorm2d(featuresGen*2),
            nn.ReLU(),

            nn.ConvTranspose2d(featuresGen*2,channelsImg,kernel_size=4,stride=2,padding=0),
            nn.Tanh()
    
        )
    def forward(self,x):
        return self.net(x)