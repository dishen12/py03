import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Aspp_b_2_mid_concat_relu(nn.Module):
    """
    串联加并联的操作的aspp,每层延伸出去，相当于一个fpn，注意，此处每层都添加了BN，没有加relu，只在最后添加了relu
    """
    def __init__(self,in_planes,out_planes,stride=1,scale=0.1,rate=[6,3,2,1]):
        #rate 1 2 5   9
        #     2 4 10  18
        #     3 6 15  27
        super(Aspp_b_2_mid_concat_relu,self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.rate = rate
        inter_planes = in_planes // 8   # 后边这个值，考虑微调 原来是8
        if(len(rate)==4):
            self.branch0 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[0], dilation=rate[0], relu=False)
            )
            self.branch0_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[0], dilation=2*rate[0], relu=False)
            self.branch0_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[0], dilation=3*rate[0], relu=False)
            self.branch1 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[1], dilation=rate[1], relu=False))
            self.branch1_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[1], dilation=2*rate[1], relu=False)
            self.branch1_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[1], dilation=3*rate[1], relu=False)  
            self.branch2 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[2], dilation=rate[2], relu=False))
            self.branch2_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[2], dilation=2*rate[2], relu=False)
            self.branch2_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[2], dilation=3*rate[2], relu=False)
            self.branch3 = nn.Sequential(
                    BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                    BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=rate[3], dilation=rate[3], relu=False))
            self.branch3_1 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*rate[3], dilation=2*rate[3], relu=False)
            self.branch3_2 = BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=3*rate[3], dilation=3*rate[3], relu=False)
            self.ConvLinear = BasicConv(24*inter_planes,out_planes,kernel_size=1,stride=1,relu=False)
            self.shortcut = BasicConv(in_planes,out_planes,kernel_size=1,stride=stride, relu=False)
            self.relu = nn.ReLU(inplace=False)
        else:
                print("error! the rate is incorrect!")
    def forward(self,x):
        # some thing there
        if(len(self.rate)==4):
            x0 = self.branch0(x)
            x0_r = self.relu(x0)
            x01 = self.branch0_1(x0_r)
            x01_r = self.relu(x01)
            x02 = self.branch0_2(x01_r)
            #print("0",x0.size(),x01.size(),x02.size())
            x1 = self.branch1(x)
            x1_r = self.relu(x1)
            x11 = self.branch1_1(x1_r)
            x11_r = self.relu(x11)
            x12 = self.branch1_2(x11_r)
            #print("1",x1.size(),x11.size(),x12.size())
            x2 = self.branch2(x)
            x2_r = self.relu(x2)
            #print("x2",x2.size())
            x21 = self.branch2_1(x2_r)
            x21_r = self.relu(x21)
            #print("x21",x21.size())
            x22 = self.branch2_2(x21_r)
            #print("x22",x22.size())
            #print("2",x2.size(),x21.size(),x22.size())
            x3 = self.branch3(x)
            x3_r = self.relu(x3)
            x31 = self.branch3_1(x3_r)
            x31_r = self.relu(x31)
            x32 = self.branch3_2(x31_r)
            #print("3",x3.size(),x31.size(),x32.size())
            
            #mid concat
            out1 = torch.cat((x0,x1,x2,x3),1)
            out1 = self.relu(out1)
            out2 = torch.cat((x01,x11,x21,x31),1)
            out2 = self.relu(out2)
            out3 = torch.cat((x02,x12,x22,x32),1)
            out3 = self.relu(out3)
            
            out = torch.cat((out1,out2,out3),1)
            #out = torch.cat((x0,x01,x02,x1,x11,x12,x2,x21,x22,x3,x31,x32),1)
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            #print("the size of shortcut is:",short.size())
            out = out*self.scale + short
            out = self.relu(out)
            return out
        else:
            print("error!")
            return 


class RFBNet(nn.Module):
    """RFB Net for object detection
    The network is based on the SSD architecture.
    Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1711.07767.pdf for more details on RFB Net.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input, size of either 300 or 512
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes,Rate=[9,5,2,1]):
        super(RFBNet, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.size = size

        if size == 300:
            self.indicator = 3
        elif size == 512:
            self.indicator = 5
        else:
            print("Error: Sorry only SSD300 and SSD512 are supported!")
            return
        # vgg network
        self.base = nn.ModuleList(base)
        # conv_4
        #self.Norm = BasicRFB_a(512,512,stride = 1,scale=1.0)
        self.Norm = Aspp_b_2_mid_concat_relu(512,512,stride=1,scale=1,rate=Rate)
        #self.aspp_a_7 = Aspp_b_2(1024,1024,stride=1,scale=1,rate=Rate)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3*batch,300,300].

        Return:
            Depending on phase:
            test:
                list of concat outputs from:
                    1: softmax layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            #print(x.size())
            x = self.base[k](x)
        
        #print("s,x",x.size())
        #s = self.Norm(x)
        s = self.Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        for k in range(23, len(self.base)):
            x = self.base[k](x)
        
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = v(x)
            if k < self.indicator or k%2 ==0:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        
        #print("the size of loc is ",len(loc))
        #print([o.size() for o in loc])


        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = (
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}


def add_extras(size, cfg, i, batch_norm=False,Rate=[6,3,2,1]):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                if in_channels == 256 and size == 512:
                    layers += [Aspp_b_2_mid_concat_relu(in_channels,cfg[k+1],stride=2,scale=1,rate=Rate)]
                else:
                    layers += [Aspp_b_2_mid_concat_relu(in_channels,cfg[k+1],stride=2,scale=1,rate=Rate)]
            else:
                layers += [Aspp_b_2_mid_concat_relu(in_channels,v,scale=1,rate=Rate)]
        in_channels = v
    if size == 512:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=4,stride=1,padding=1)]
    elif size ==300:
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
        layers += [BasicConv(256,128,kernel_size=1,stride=1)]
        layers += [BasicConv(128,256,kernel_size=3,stride=1)]
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return
    return layers

extras = {
    '300': [1024, 'S', 512, 'S', 256],
    '512': [1024, 'S', 512, 'S', 256, 'S', 256,'S',256],
}


def multibox(size, vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [-2]
    for k, v in enumerate(vgg_source):
        if k == 0:
            loc_layers += [nn.Conv2d(512,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers +=[nn.Conv2d(512,
                                 cfg[k] * num_classes, kernel_size=3, padding=1)]
        else:
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    i = 1
    indicator = 0
    if size == 300:
        indicator = 3
    elif size == 512:
        indicator = 5
    else:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    for k, v in enumerate(extra_layers):
        if k < indicator or k%2== 0:
            loc_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                 * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[i]
                                  * num_classes, kernel_size=3, padding=1)]
            i +=1
    return vgg, extra_layers, (loc_layers, conf_layers)

mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
}


def build_net(phase, size=300, num_classes=21,rate="6,3,2,1"):
    Rate = [int(i) for i in rate.strip().split(",")]
    print("the rate is ",Rate)
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 300 and size != 512:
        print("Error: Sorry only RFBNet300 and RFBNet512 are supported!")
        return

    return RFBNet(phase, size, *multibox(size, vgg(base[str(size)], 3),
                                add_extras(size, extras[str(size)], 1024,Rate=Rate),
                                mbox[str(size)], num_classes), num_classes,Rate)
