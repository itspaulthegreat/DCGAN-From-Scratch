import torch
import torch.nn as nn




class Discriminator(nn.Module):
    def __init__(self,img_channels,features_d):
        super(Discriminator,self).__init__()
        
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self.layerblock(features_d,2*features_d,kernel_size=4,stride=2,padding=1),            
            self.layerblock(2*features_d,4*features_d,kernel_size=4,stride=2,padding=1),
            self.layerblock(4*features_d,8*features_d,kernel_size=4,stride=2,padding=1),
            nn.Conv2d(8*features_d,1,kernel_size=4,stride=2,padding=0),
            nn.Sigmoid()
        )

    def layerblock(self,in_channels,out_channels,kernel_size,stride,padding):

        layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

        return layer

    def forward(self,x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self,z_dim,img_channels,features_g):
        super(Generator,self).__init__()
        #input N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self.layerblock(z_dim,features_g*16,kernel_size=4,stride=1,padding=0),
            self.layerblock(features_g *16,8*features_g,kernel_size=4,stride=2,padding=1),            
            self.layerblock(8*features_g,4*features_g,kernel_size=4,stride=2,padding=1),
            self.layerblock(4*features_g,2*features_g,kernel_size=4,stride=2,padding=1),
            nn.ConvTranspose2d(2*features_g,img_channels,kernel_size=4,stride=2,padding=1),
            nn.Tanh()
        )
    def layerblock(self,in_channels,out_channels,kernel_size,stride,padding):

        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        return layer

    def forward(self,x):
        return self.gen(x)
    
    
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data,0.0,0.02)

