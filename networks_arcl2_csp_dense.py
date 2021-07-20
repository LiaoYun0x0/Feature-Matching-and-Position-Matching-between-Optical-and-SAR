import torch
import torch.nn as nn
import torch.nn.functional as f
import math
from new_networks import std_conv_embede, csp_resnet_embede, densenet_embede, csp_densenet_embede
import time
import numpy as np

def l2_norm(input,axis=-1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

class EmbeddingNet(nn.Module):
    def __init__(self,dim_desc=256,drop_rate=0.1):
        super(EmbeddingNet, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        
        #self.embede = std_conv_embede()
        #self.embede = csp_resnet_embede()
        #self.embede = densenet_embede()
        self.embede = csp_densenet_embede()

    def forward(self, patch):
        descr = l2_norm(self.embede(patch).view(-1,self.dim_desc))
        return descr

class EmbeddingNet_SARptical(nn.Module):
    def __init__(self,dim_desc=256,drop_rate=0.1):
        super(EmbeddingNet_SARptical, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        
        #self.embede = std_conv_embede()
        #self.embede = csp_resnet_embede()
        #self.embede = densenet_embede()
        self.embede = csp_densenet_embede(block_config=[2,2,2,2])

    def forward(self, patch):
        descr = l2_norm(self.embede(patch).view(-1,self.dim_desc))
        return descr

class EmbeddingNet_ROIsummer(nn.Module):
    def __init__(self,dim_desc=256,drop_rate=0.1):
        super(EmbeddingNet_ROIsummer, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.eps_fea_norm = 1e-5
        
        #self.embede = std_conv_embede()
        #self.embede = csp_resnet_embede()
        #self.embede = densenet_embede()
        self.embede = csp_densenet_embede(block_config=[2,2,2,2,2])

    def forward(self, patch):
        descr = l2_norm(self.embede(patch).view(-1,self.dim_desc))
        return descr



class LDMNet(nn.Module):
    def __init__(self,EmbeddingNet,m=0.5):
        super(LDMNet,self).__init__()
        self.EmbeddingNet = EmbeddingNet
        self.sin_m = math.sin(m)
        self.mm = self.sin_m*m
        self.cos_m = math.cos(m)
        self.thresh = math.cos(math.pi - m)
        self.weight_decay = 5e-4
        self.s = 64
        # self.alpha = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta = 0.

    def get_embedding(self,x):
        return self.EmbeddingNet(x)
    def forward(self,p,a,label): # positive batch, anchor batch 
        pe = self.EmbeddingNet(p)
        ae = self.EmbeddingNet(a)
        angles = self.arc(pe,ae,label)
        loss_l2,_ = self.loss_l2_hard(pe,ae,sos=True)
        #label = torch.nn.functional.softmax(label,dim=-1)
        loss_arc = self.CrossEntropyLoss(angles,label)
        # print('l2 loss %f'%loss_l2)
        # print('arc loss %f'%loss_arc)
        # m = torch.sigmoid(self.alpha).cuda()
        # n = torch.sigmoid(self.beta).cuda()
        t_loss = loss_l2 + self.beta*loss_arc
        return t_loss,angles,pe,ae

    def CrossEntropyLoss(self,logit,label):
        # CrossEntropyLoss for one-hot label 
        logprobs = torch.nn.functional.log_softmax (logit, dim = 1)
        return  -(label * logprobs).sum() / logit.shape[0]

    def arc(self,pe,ae,label):
        bs,f = pe.size()
        mask0 = (torch.eye(bs) == 0)
        angles1 = pe.matmul(ae.transpose(0,1))
        angles2 = ae.matmul(pe.transpose(0,1))[mask0].view(bs,bs-1)
        angles = torch.cat([angles1,angles2],dim=-1)
        # true angles: (bs,2*bs-1)
        angles = angles.clamp(-1,1)
        # cos_pa: gt_match patch cos value (which label=1)
        cos_pa = torch.sum(pe*ae,axis=-1)
        cos_pa = cos_pa.clamp(-1,1)
        sin_pa = torch.sqrt(1 - torch.pow(cos_pa,2))
        # cos_pa_m = cos(pa+m)
        cos_pa_m = cos_pa*self.cos_m - sin_pa*self.sin_m
        cosface = cos_pa - self.mm
        # if -m < angle_pa < pi-m, use arcface (cost = cos(t+m))
        # else use cosface (cost = cost-f(m))
        mask = cos_pa-self.thresh <= 0 
        cos_pa_m[mask] = cosface[mask]
        angles[label==1] = cos_pa_m
        return angles*self.s


    def loss_l2_hard(self,pe,ae,margin=1,sos=False):
        d_pa = self.distance_pa(pe,ae)
        d_n,sos_item = self.distance_n(pe,ae,sos=sos)
        diff = f.relu(d_pa-d_n+margin).squeeze()
        loss = torch.sum(torch.pow(diff,2))/pe.shape[0] + sos_item
        return loss,diff

    def distance_pa(self,pe,ae):
        dist = torch.sqrt(torch.sum(torch.pow(pe-ae,2),dim=-1,keepdim=True))
        return dist

    def distance_n(self,ae,pe,k=1,sos=False):
        b,f = pe.size()
        mask = torch.eye(b).view(-1) == 0
        expand_pe1 = (pe.expand(b,b,f).reshape(-1,f))[mask,:] # [b*(b-1),f]
        expand_pe2 = (pe.reshape(b,1,f).expand(b,b,f).reshape(-1,f))[mask,:] # [b*(b-1),f]
        expand_ae1 = (ae.expand(b,b,f).reshape(-1,f))[mask,:]
        expand_ae2 = (ae.reshape(b,1,f).expand(b,b,f).reshape(-1,f))[mask,:]
        dist_pe1_ae2 = self.distance_pa(expand_pe1,expand_ae2).reshape(b,b-1)
        dist_pe2_ae1 = self.distance_pa(expand_pe2,expand_ae1).reshape(b,b-1)
        d_n = torch.cat([dist_pe1_ae2,dist_pe2_ae1],dim=-1)
        #d_n,_ = torch.min(d_n,dim=-1,keepdim=True) # (b,)
        d_n,_ = torch.topk(d_n,dim=-1,k=1,largest=False,sorted=False)
        sos_val = 0
        if sos:
            dist_pe = self.distance_pa(expand_pe1,expand_pe2).reshape(b,b-1) # (b,b-1)
            dist_ae = self.distance_pa(expand_ae1,expand_ae2).reshape(b,b-1)
            diff = torch.pow(dist_pe-dist_ae,2) #(b,b-1)
            _,index_pe = torch.topk(dist_pe,k=k,dim=-1,largest=False,sorted=False) # (b,k)
            _,index_ae = torch.topk(dist_ae,k=k,dim=-1,largest=False,sorted=False) 
            index = torch.cat([index_pe,index_ae],dim=-1) # (b,2k)
            sos_val = 0
            for i in range(b):
                sos_index = (index[i]).unique()
                sos_val += torch.sqrt(torch.sum(diff[i,sos_index]))
            sos_val = sos_val / b
            #sos_val = sos_val *self.weight_decay
        return d_n,sos_val


    def angle_summary(self,p,a,label):
        pe = self.EmbeddingNet(p)
        ae = self.EmbeddingNet(a)
        bs,f = pe.size()
        mask0 = (torch.eye(bs) == 0)
        angles1 = pe.matmul(ae.transpose(0,1))
        angles2 = ae.matmul(pe.transpose(0,1))[mask0].view(bs,bs-1)
        angles = torch.cat([angles1,angles2],dim=-1)
        # print(angles)
        # print(angles.shape)
        # print(torch.sum(label))
        # print(label.shape[0]*label.shape[1]-torch.sum(label))
        angle_pos = angles[label==1]
        angle_neg = angles[label==0]
        # theta_pos = np.arccos(angle_pos.cpu().detach().numpy())*180/np.pi
        # theta_neg = (np.arccos(angle_neg.cpu().detach().numpy())*180)/np.pi

        # plot img, choose first one
        return list(angle_pos.cpu().detach().numpy()), list(angle_neg.cpu().detach().numpy())
        # eval only 
        # return angle_pos, angle_neg

if __name__ == '__main__':
    '''
    利用训练好的网络，对测试集样本间的余弦距离分布进行统计，并显示分布直方图
    '''
    # torch.cuda.set_device('cuda:1')
    import cv2
    from tqdm import tqdm
    from glob import glob
    from old.utils import RocketsDataset
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from train_arcl2_cspdense_ROIpatch import ap_comput
    # from old.networks_angle_cls_csp_dense import EmbeddingNet,LDMNet
    embeddingNet = EmbeddingNet()
    embeddingNet.eval()
    embeddingNet.cuda()
    ldmNet = LDMNet(embeddingNet)
    ldmNet.eval()
    ldmNet.cuda()
    img_path = 'old/ROIs1868_summer/testpatch.txt'

    pathes = glob('weights_arcl2_cspdense_ROIpatch/'+'*.tar')
    pathes = sorted(pathes)
    pathes = ['old/weights/weights_agl_cls_cspdense_ROIpatch/rockets_62_2740_540_317.692.tar']
    total_ap = []
    for model_path in pathes:

        checkpoint = torch.load(model_path,map_location='cuda:0')
        # checkpoint = torch.load('old/weights/weights_agl_cls_cspdense_ROIpatch/rockets_62_2740_540_317.692.tar',map_location='cuda:0')
        ldmNet.load_state_dict(checkpoint['model_state_dict']) 



        test_loader = DataLoader(RocketsDataset(img_path, transform=None, mode='eval'),
                                batch_size=512, shuffle=False, num_workers=0)

        p_x = []
        angles_pos=[]
        angles_neg = []
        for i,sample in enumerate(tqdm(test_loader)):
            p_x.append(i)
            optical, sar, = sample['optical'].cuda().float(), sample['sar'].cuda().float()
            label = torch.eye(optical.shape[0]).cuda().float()
            v2 = torch.zeros(optical.shape[0],optical.shape[0]-1).cuda().float()
            label = torch.cat([label,v2],dim=-1)
            with torch.no_grad():

        
                angle_pos,angle_neg = ldmNet.angle_summary(optical,sar,label)
            # print(len(angle_neg))
            angles_pos.extend(angle_pos)
            angles_neg.extend(angle_neg)
            
        print('###########')
        print(model_path)
        ap = ap_comput(np.array(angles_pos),np.array(angles_neg))
        total_ap.append(ap)
        print(ap)

    print('@@@@@@@@@@@@@')
    max_ap = max(total_ap)
    print(max_ap)
    print(pathes[total_ap.index(max_ap)])
    plt.hist(angles_pos,bins=100,alpha=0.5,label='pos')
    plt.hist(angles_neg[:int(2*len(angles_pos))],bins=100,alpha=0.5,label='neg')
    plt.legend()
    plt.savefig('hist_angles.png',dpi=600)
