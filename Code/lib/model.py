import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F
from Code.lib.res2net_v1b_base import Res2Net_model

def maxpool():
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    return pool


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x



#Global Contextual module
class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



###############################################################################

class CIM0(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CIM0, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        

        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        

    def forward(self, rgb, depth):
        
        ################################
        
        x_rgb = self.layer_10(rgb)
        x_dep = self.layer_20(depth)
        
        rgb_w = nn.Sigmoid()(x_rgb)
        dep_w = nn.Sigmoid()(x_dep)
        
        ##
        x_rgb_w = rgb.mul(dep_w)
        x_dep_w = depth.mul(rgb_w)
        
        x_rgb_r = x_rgb_w + rgb
        x_dep_r = x_dep_w + depth
        
        ## fusion 
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)         
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
         
        return out1


class CIM(nn.Module):    
    def __init__(self,in_dim, out_dim):
        super(CIM, self).__init__()
        
        act_fn = nn.ReLU(inplace=True)
        
        self.reduc_1 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        self.reduc_2 = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1), act_fn)
        
        self.layer_10 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)   
        
        self.layer_11 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)        
        self.layer_21 = nn.Sequential(nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        

        self.layer_ful1 = nn.Sequential(nn.Conv2d(out_dim*2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)
        self.layer_ful2 = nn.Sequential(nn.Conv2d(out_dim+out_dim//2, out_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(out_dim),act_fn,)

    def forward(self, rgb, depth, xx):
        
        ################################
        x_rgb = self.reduc_1(rgb)
        x_dep = self.reduc_2(depth)
        
        x_rgb1 = self.layer_10(x_rgb)
        x_dep1 = self.layer_20(x_dep)
        
        rgb_w = nn.Sigmoid()(x_rgb1)
        dep_w = nn.Sigmoid()(x_dep1)
        
        ##
        x_rgb_w = x_rgb.mul(dep_w)
        x_dep_w = x_dep.mul(rgb_w)
        
        x_rgb_r = x_rgb_w + x_rgb
        x_dep_r = x_dep_w + x_dep
        
        ## fusion 
        x_rgb_r = self.layer_11(x_rgb_r)
        x_dep_r = self.layer_21(x_dep_r)
        
        
        ful_mul = torch.mul(x_rgb_r, x_dep_r)         
        x_in1   = torch.reshape(x_rgb_r,[x_rgb_r.shape[0],1,x_rgb_r.shape[1],x_rgb_r.shape[2],x_rgb_r.shape[3]])
        x_in2   = torch.reshape(x_dep_r,[x_dep_r.shape[0],1,x_dep_r.shape[1],x_dep_r.shape[2],x_dep_r.shape[3]])
        x_cat   = torch.cat((x_in1, x_in2),dim=1)
        ful_max = x_cat.max(dim=1)[0]
        ful_out = torch.cat((ful_mul,ful_max),dim=1)
        
        out1 = self.layer_ful1(ful_out)
        out2 = self.layer_ful2(torch.cat([out1,xx],dim=1))
         
        return out2



class MFA(nn.Module):    
    def __init__(self,in_dim):
        super(MFA, self).__init__()
         
        self.relu = nn.ReLU(inplace=True)
        
        self.layer_10 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)
        self.layer_20 = nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1)   
        self.layer_cat1 = nn.Sequential(nn.Conv2d(in_dim*2, in_dim, kernel_size=3, stride=1, padding=1),nn.BatchNorm2d(in_dim),)        
        
    def forward(self, x_ful, x1, x2):
        
        ################################
    
        x_ful_1 = x_ful.mul(x1)
        x_ful_2 = x_ful.mul(x2)
        
     
        x_ful_w = self.layer_cat1(torch.cat([x_ful_1, x_ful_2],dim=1))
        out     = self.relu(x_ful + x_ful_w)
        
        return out
    
    

  
   
###############################################################################

class SPNet(nn.Module):
    def __init__(self, channel=32,ind=50):
        super(SPNet, self).__init__()
        
       
        self.relu = nn.ReLU(inplace=True)
        
        self.upsample_2 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.upsample_4 = nn.Upsample(scale_factor=0.25, mode='bilinear', align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=0.125, mode='bilinear', align_corners=True)

        #Backbone model
        #Backbone model
        self.layer_rgb  = Res2Net_model(ind)
        
        
        ###############################################
        # decoders #
        ###############################################
        
        ## rgb
        self.rgb_gcm_4    = GCM(2048,  channel)
        
        self.rgb_gcm_3    = GCM(1024+32,  channel)

        self.rgb_gcm_2    = GCM(512+32,  channel)

        self.rgb_gcm_1    = GCM(256+32,  channel)

        self.rgb_gcm_0    = GCM(64+32,  channel)        
        #full size
#         self.rgb_conv_out = nn.Conv2d(channel, 20, kernel_size=7, stride=4, padding=0)
        self.rgb_conv_out = nn.Conv2d(channel, 20, kernel_size=5, stride=1, padding=0)

        
               

    def forward(self, imgs):
        
        img_0, img_1, img_2, img_3, img_4 = self.layer_rgb(imgs)
        
        
        ####################################################
        ## decoder rgb
        ####################################################        
        #
        x_rgb_42    = self.rgb_gcm_4(img_4)
        
        x_rgb_3_cat = torch.cat([self.upsample_2(img_3), x_rgb_42], dim=1)
        x_rgb_32    = self.rgb_gcm_3(x_rgb_3_cat)
        
        x_rgb_2_cat = torch.cat([self.upsample_4(img_2), x_rgb_32], dim=1)
        x_rgb_22    = self.rgb_gcm_2(x_rgb_2_cat)        

        x_rgb_1_cat = torch.cat([self.upsample_8(img_1), x_rgb_22], dim=1)
        x_rgb_12    = self.rgb_gcm_1(x_rgb_1_cat)   

        x_rgb_0_cat = torch.cat([self.upsample_8(img_0), x_rgb_12], dim=1)
        x_rgb_02    = self.rgb_gcm_0(x_rgb_0_cat)   


        rgb_out     = self.upsample_4(self.rgb_conv_out(x_rgb_02))


#         rgb_out     = self.rgb_conv_out(self.upsample_4(x_rgb_02))

        


        return rgb_out
    
    

    def _make_agant_layer(self, inplanes, planes):
        layers = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )
        return layers

    def _make_transpose(self, block, planes, blocks, stride=1):
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes,
                                   kernel_size=2, stride=stride,
                                   padding=0, bias=False),
                nn.BatchNorm2d(planes),
            )
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
    
   

 

if __name__ == '__main__':
    model = SPNet()
    image  = torch.randn(1, 3, 1536, 2048) #full size
    # image  = torch.randn(1, 3, 384, 512)
    output = model(image)
    print(output.size())
