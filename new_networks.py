import torch
import torch.nn as nn
import torch.nn.functional as f
import math

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,padding=k//2, dilation=d, groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c2)
        #self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()
        self.act = nn.PReLU(c2) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Conv_inv(nn.Module):
    # densenet use bn+act+conv architecture
    def __init__(self, c1, c2, k=1, s=1, d=1, g=1):
        super(Conv_inv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s,padding=k//2, dilation=d, groups=g,bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.PReLU(c1)
    def forward(self, x):
        return self.conv(self.act(self.bn(x)))

class Denselayer(nn.Module):
    def __init__(self,cin,growth,bn_size):
        super(Denselayer,self).__init__()
        c_ = int(growth * bn_size)
        self.conv1 = Conv_inv(cin,c_,1,1)
        self.conv2 = Conv_inv(c_,growth,3,1)
    def forward(self,x):
        feature = self.conv2(self.conv1(x))
        return torch.cat([x,feature],1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers,in_channels,bn_size,growth):
        super(DenseBlock,self).__init__()
        self.block = nn.Sequential()
        for i in range(num_layers):
            self.block.add_module('layer_%d'%(i),Denselayer(in_channels+growth*i,growth,bn_size))
    def forward(self,x):
        return self.block(x)

class DenseBlockCSP(nn.Module):
    def __init__(self, num_layers,in_channels,bn_size,growth):
        super(DenseBlockCSP,self).__init__()
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.bn_size = bn_size
        self.growth = growth
        self.block = self._make_block()
        self.trans = self._make_trans() 

    def _make_block(self):
        _r = self.in_channels - self.in_channels // 2
        _block = nn.Sequential()
        for i in range(self.num_layers):
            _block.add_module('layer_%d'%(i),Denselayer(_r+self.growth*i,self.growth,self.bn_size))
        return _block

    def _make_trans(self):
        _l = self.in_channels // 2
        _r = self.in_channels - _l
        _c = _r + self.growth * self.num_layers
        trans = Conv_inv(_c,_c//2)
        return trans

    def forward(self,x):
        split = self.in_channels //2 # for (B,C,H,W) data format 
        lpart = x[:,:split,...]
        rpart = x[:,split:,...]
        feature = self.trans(self.block(rpart))
        return torch.cat([lpart,feature],1)


class Transition(nn.Module):
    def __init__(self,cin,cou):
        super(Transition,self).__init__()
        self.conv = Conv_inv(cin,cou,1,1)
        self.pool = nn.AvgPool2d(2,2)
    def forward(self,x):
        return self.pool(self.conv(x))

class DenseNet(nn.Module):
    def __init__(self,block_config=[3,4,5],init_channel=32,bn_size=4,growth=12):
        super(DenseNet,self).__init__()
        _c = init_channel
        self.conv = Conv_inv(1,_c,3,2)
        self.net = nn.Sequential()
        for i,num_layers in enumerate(block_config):
            self.net.add_module('block_%d'%(i),DenseBlock(num_layers,_c,bn_size,growth))
            _c = int(_c + growth * num_layers)
            if i < len(block_config)-1:
                self.net.add_module('transition_%d'%(i),Transition(_c,int(_c//2)))
                _c = int(_c // 2)
        self.cou = _c
    def forward(self,x):
        return self.net(self.conv(x))

class DenseNetCSP(nn.Module):
    def __init__(self,block_config=[3,4,5],init_channel=32,bn_size=4,growth=12):
        super(DenseNetCSP,self).__init__()
        _c = init_channel
        self.conv = Conv_inv(1,_c,3,2)
        self.net = nn.Sequential()
        for i,num_layers in enumerate(block_config):
            self.net.add_module('block_%d'%(i),DenseBlockCSP(num_layers,_c,bn_size,growth))
            _l = _c // 2
            _r = _c - _l
            _c = _l + int((_r + num_layers * growth)//2)
            if i < len(block_config)-1:
                self.net.add_module('transition_%d'%(i),Transition(_c,int(_c//2)))
                _c = int(_c // 2)
        self.cou = _c
    def forward(self,x):
        x = self.conv(x)
        x = self.net(x)
        return x
   

class BottleNeck(nn.Module):
    def __init__(self,c1,c2, k=3,s=1,act=True,e=0.5):
        super(BottleNeck,self).__init__()
        c_ = int(c2*e)
        self.conv1 = Conv(c1,c_,1,1)
        self.conv2 = Conv(c_,c2,k,s,act=False)
        if c1==c2 and s==1:
            self.conv3 = nn.Identity()
        else:
            self.conv3 = Conv(c1,c2,1,s,act=False)
        self.act = nn.PReLU(c2)
    def forward(self,x):
        return self.act(self.conv3(x) + self.conv2(self.conv1(x)))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(c2, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[BottleNeck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

    
def std_conv_embede(drop_rate=0.1,dim_desc=256):
    embede = nn.Sequential(
            nn.InstanceNorm2d(1),
            Conv(1,32,3,1),
            Conv(32,32,3,2),#d2
            Conv(32,32,1,1),
            Conv(32,64,3,1), 
            Conv(64,64,3,2), #d4
            Conv(64,64,1,1),
            Conv(64,128,3,1), 
            Conv(128,128,3,2), #d8
            Conv(128,128,1,1),
            nn.Dropout(drop_rate),
            nn.Conv2d(128,dim_desc, 8,1),
            nn.BatchNorm2d(dim_desc)
        )
    return embede

def csp_resnet_embede(drop_rate=0.1,dim_desc=256):
    embede = nn.Sequential(
            nn.InstanceNorm2d(1),
            Conv(1,32,3,2), # d2
            BottleneckCSP(32,32),
            Conv(32,64,3,2), # d4
            BottleneckCSP(64,64),
            Conv(64,128,3,2), #d8
            BottleneckCSP(128,128),
            nn.Dropout(drop_rate),
            nn.Conv2d(128,dim_desc,8,1),
            nn.BatchNorm2d(dim_desc)
        )
    return embede

def densenet_embede(drop_rate=0.1,dim_desc=256,block_config=[2,2,2]):
    densenet = DenseNet(block_config=block_config,init_channel=32,bn_size=4,growth=12)
    out_channel = densenet.cou
    print(out_channel)
    embede = nn.Sequential(
        nn.InstanceNorm2d(1),
        densenet,
        nn.BatchNorm2d(out_channel),
        nn.PReLU(out_channel),
        nn.Dropout(drop_rate),
        nn.Conv2d(out_channel,dim_desc,8,1),
        nn.BatchNorm2d(dim_desc)
    )
    return embede

def csp_densenet_embede(drop_rate=0.1,dim_desc=256,block_config=[2,2,2]):
    densenetCSP = DenseNetCSP(block_config=block_config,init_channel=32,bn_size=4,growth=12)
    out_channel = densenetCSP.cou
    print("csp output_channel:",out_channel)
    embede = nn.Sequential(
        nn.InstanceNorm2d(1),
        densenetCSP,
        nn.BatchNorm2d(out_channel),
        nn.PReLU(out_channel),
        nn.Dropout(drop_rate),
        nn.Conv2d(out_channel,dim_desc,8,1),
        nn.BatchNorm2d(dim_desc)
    )
    return embede

if __name__ == "__main__":
    #net = DenseBlockCSP(num_layers=3,in_channels=32,bn_size=4,growth=12).cuda()
    net = csp_resnet_embede().cuda()
    x = torch.rand(32,1,64,64).cuda().float()
    y = net(x)
    print(y.shape)