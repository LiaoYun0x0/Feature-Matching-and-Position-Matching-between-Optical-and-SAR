'''

该文件用于ROI_summer 数据集上,192 大小对256大小的sar-optical图像匹配,
统计匹配位置误差,匹配FLAG(<5 PIXELS), 匹配关键点数量, 关键点误差(<= 2 PIXELS)
related results are saved in eval_results_XX.txt

'''

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import math
import os
import time
from glob import glob


class models():
    def __init__(self,dataset,weights_list):
        self.dataset = dataset
        self.weights_list - weights_list




def model_choose(model_path):

    if 'agl_cls_cspdense' in model_path:

        from old.networks_angle_cls_csp_dense import EmbeddingNet, LDMNet
        embede_model = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using agl_cls_cspdense net')
        return embede_model,ldm_model

    elif 'HardNet' in model_path:

        from old.networks_l2_sos import EmbeddingNet, LDMNet
        embede_model = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using Hardnet net')

        return embede_model,ldm_model

    elif 'classification' in model_path:
        from old.networks_angle_classification import EmbeddingNet, LDMNet
        embede_model = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using classification simple net')

        return embede_model,ldm_model

    elif 'TFeat' in model_path:
        from old.networks_tfeat import TNet_Rocket, LDMNet
        embede_model = TNet_Rocket()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using Tfeat net')

        return embede_model,ldm_model

    elif 'MatchNet' in model_path:
        from old.network_matchnet_512 import FeatureNet, CLSNet
        embede_model = FeatureNet()
        embede_model.cuda()
        ldm_model = CLSNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using matchnet net...')

        return embede_model,ldm_model

    elif '_l2_cspdense' in model_path:
        from old.networks_l2_cls_csp_dense import EmbeddingNet, LDMNet
        embede_model  = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using l2_cspdense net...')
        return embede_model,ldm_model

    elif 'entropy_cspdense' in model_path:
        from old.networks_entropyloss_cls_csp_dense import EmbeddingNet, CLSNet
        embede_model  = EmbeddingNet()
        embede_model.cuda()
        ldm_model = CLSNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using entropy_csp_dense net...')
        return embede_model,ldm_model

    elif 'agl_dense' in model_path:
        from old.networks_angle_cls_dense import EmbeddingNet,LDMNet
        embede_model  = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using agl_dense net...')
        return embede_model,ldm_model

    elif 'cspdense64_joint' in model_path:
        from old.networks_l2_cls_csp_dense import EmbeddingNet
        from joint_model.jointNet import JointNet,LDMNet
        embede_model  = JointNet(EmbeddingNet())
        embede_model.cuda()
        ldm_model = LDMNet(embede_model).cuda().float()
        # ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using cspdense64_joint net...')
        global joint_flag 
        joint_flag = True
        return embede_model,ldm_model

    elif 'arcl2_cspdense' in model_path:
        from networks_arcl2_csp_dense import EmbeddingNet,LDMNet
        embede_model  = EmbeddingNet()
        embede_model.cuda()
        ldm_model = LDMNet(embede_model)
        ldm_model.cuda().eval()
        load_model(ldm_model,model_path)
        print('using arcl2 cspdense net...')
        return embede_model,ldm_model



def load_model(model,model_path):
    checkpoint = torch.load(model_path,map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict']) 
    return 

def kpts2descriptors(kpts,img,model,batch_size=128,patch_size=64,use_gpu=True):
    descrs = []
    length = len(kpts)
    shards = int(np.ceil(length/batch_size))
    h,w = img.shape[:2]
    sx,sy,ex,ey = 0,0,0,0
    ltoffsets = patch_size//2
    rboffsets = patch_size-ltoffsets
    for i in range(shards):
        patches = []
        batch_kpts = kpts[i*batch_size:min((i+1)*batch_size,length)]
        for kp in batch_kpts:
            x, y = kp.pt
            x = int(x)
            y = int(y)
            if x <= ltoffsets:
                sx = 0
                ex = sx + patch_size
            else:
                ex = min(x+rboffsets,w)
                sx = ex-patch_size
            if y <= ltoffsets:
                sy = 0
                ey = sy + patch_size
            else:
                ey = min(y+rboffsets,h)
                sy = ey-patch_size
            patch = img[int(sy):int(ey),int(sx):int(ex)]
            assert patch.shape[0] == patch_size and patch.shape[1] == patch_size,str(patch.shape)+'  '+str(x)+' '+str(y)
            patches.append(patch)

        patches = torch.from_numpy(np.asarray(patches)).float()
        patches = torch.unsqueeze(patches, 1)
        if use_gpu:
            patches = patches.cuda()

        descrs.append(model(patches).detach().cpu().numpy())

    descrs = np.concatenate(descrs,axis=0)
    # print(len(descrs))
    return descrs

def matches2offsets(matches,queryKp,trainKp):
    offsets = []
    for m in matches:
        train_x,train_y = trainKp[m.trainIdx].pt
        train_x,train_y = round(train_x),round(train_y)
        query_x,query_y = queryKp[m.queryIdx].pt
        query_x,query_y = round(query_x),round(query_y)
        offset_x = query_x - train_x
        offset_y = query_y - train_y
        if 0 < offset_x < 289 and 0 < offset_y < 289:
            offsets.append([offset_x,offset_y])
    return offsets

def matched_kpt_summary(offsets,gt_point):
    # 统计匹配上的关键点数量和kpt平均误差
    matched_cnt = 0
    matched_diff = 0.
    diff = np.array(offsets) - np.array(gt_point)
    distance = np.sqrt(diff[:,0]*diff[:,0]+diff[:,1]*diff[:,1])
    matched_cnt = np.sum(distance <= 2)
    mask = (distance <= 2).reshape(-1,1)
    if matched_cnt != 0:
        matched_diff = np.mean(distance[distance <= 2])
        mask = np.concatenate([mask,mask],1)
        diff = abs(diff[mask].reshape(-1,2))
    else:
        matched_diff = 0
        diff = np.zeros([1,2])
    # print(distance[distance <= 2])
    # print('match cnt%f, matched diff%f'%(matched_cnt,matched_diff))
    return matched_cnt, matched_diff, [np.mean(diff[:,0]),np.mean(diff[:,1])]
        


def match_images(img1,img2,model,fp_detector,MIN_MATCH_COUNT=4,homo=True,thresh=1.20,knn=1,gt_point=[0,0]):
    normalize_img1 = cv2.normalize(img1,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX)
    normalize_img2 = cv2.normalize(img2,dst=None,alpha=450,beta=10,norm_type=cv2.NORM_MINMAX)
    # normalize_img1 = img1
    # normalize_img2 = img2
    kp1 = fp_detector.detect(normalize_img1, None)
    kp2 = fp_detector.detect(normalize_img2, None)
    desc_tfeat1 = kpts2descriptors(kp1,img1,model)
    desc_tfeat2 = kpts2descriptors(kp2,img2,model)
    #print('query.shape:',desc_tfeat1.shape)
    #print('train.shape:',desc_tfeat2.shape)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    
    matches = bf.knnMatch(desc_tfeat1,desc_tfeat2, k=knn)
    good = []
    for m in matches:
        for mm in m:
            if mm.distance < thresh:
                good.append(mm)
    #print('num good matches:',len(good))

    # src_pts = np.float32([ kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    # dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    # offsets_k,flag = matches2offsets_v2(src_pts,dst_pts)

    offsets = matches2offsets(good,kp1,kp2)
    matched_cnt, matched_diff = 0. ,0.
    diff_xy = [0,0]
    length = len(offsets)
    if length >= 2:
        matched_cnt, matched_diff, diff_xy = matched_kpt_summary(offsets,gt_point)
        
    if length == 0:
        offsets = [144,144]
    elif length <= 2:
        offsets = offsets[0]
    else:
        #offsets = kmeans(offsets)
        offsets = find_most_common(offsets)
    
    good_temp = []
    px,py = 0,0
    for m in good:
        qx,qy = kp1[m.queryIdx].pt
        #qx,qy = round(qx),round(qy)
        tx,ty = kp2[m.trainIdx].pt
        #tx,ty = round(tx),round(ty)
        if abs(qx-tx - offsets[0]) < 1 and abs(qy-ty- offsets[1]) < 1:
            good_temp.append(m)
            px += (qx-tx)
            py += (qy-ty)
    good = good_temp

    match_img = cv2.drawMatches(img1,kp1,img2,kp2,good,None,(0,0,255),flags=2)
    bk_gd = np.ones((match_img.shape))
    bk_gd[0:img1.shape[1],0:img1.shape[1]] = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    bk_gd[0:img2.shape[1],img1.shape[1]:(img1.shape[1]+img2.shape[1])] = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    bk_gd = cv2.rectangle(bk_gd,(offsets[0],offsets[1]),(int(offsets[0]+img2.shape[1]),int(offsets[1]+img2.shape[1])),(0,0,255),3)
    
    return offsets,[match_img,bk_gd],[matched_cnt,matched_diff,diff_xy]

def find_most_common(offsets):
    kernel = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            kernel[i,j] = 1 - math.sqrt((i-3)**2+(j-3)**2)/5

    # cv2.imshow('kernel',kernel)
    # cv2.waitKey(0)
    array = np.zeros((289,289))
    for x,y in offsets:
        ksy,key,ksx,kex = 0,7,0,7
        asy = y-3
        aey = y + 4
        asx = x-3
        aex = x+4
        if asy < 0:
            asy = 0
            ksy = 3-y
        if aey > 289:
            key = 7+289-aey
            aey = 289
        if asx < 0:
            asx = 0
            ksx = 3-x
        if aex > 289:
            kex = 7+289-aex
            aex = 289
        #print(sy,ey,sx,ex)

        array[asy:aey,asx:aex] += kernel[ksy:key,ksx:kex]
    index = array.reshape(-1).argmax()
    offsets = [index%289,index//289]
    return offsets

    
def validate(model,fp_detector,data_path,save_path):

    match_count = 0
    match_error = 0
    error_x = 0
    error_y = 0
    kpt_cnt, kpt_diff = [],[]
    cnt = 1
    opt_imgs_path = glob(data_path+'/opt_*')
    for i,opt_path in enumerate(opt_imgs_path):
        # try:
        x,y = opt_path.split('/')[-1].split('.')[0].split('_')[2:]
        x,y = float(x),float(y)
        sar_path = opt_path.replace('opt','sar')
        opt_img = cv2.imread(opt_path,0)
        sar_img = cv2.imread(sar_path,0)
        offsets,result_img, matched_info = match_images(opt_img,sar_img,model,fp_detector,homo=False,thresh=7,knn=2,gt_point=[x,y]) # use opt as queryImg, sar as trainImg
        
        distance = math.sqrt(math.pow(offsets[0]-x,2) + math.pow(offsets[1]-y,2))
        match_flag =  distance < 5
        kpt_cnt.append(matched_info[0])
        kpt_diff.append(matched_info[1])
        if match_flag:
            error_x += abs(offsets[0] - x)
            error_y += abs(offsets[1] - y)
            match_error += distance
            match_count += 1

        cv2.imwrite('./%d_%.2f_%d_kset.png'%(i,distance,match_flag),result_img[0])
        cv2.imwrite('./%d_%.2f_%d_kset_circle.png'%(i,distance,match_flag),result_img[1])
        cnt += 1

    kpt_avg_cnt = np.mean(np.array(kpt_cnt))
    kpt_avg_diff = np.mean(np.array(kpt_diff))

    return match_count,match_error/match_count, error_x/match_count, error_y/match_count,[kpt_avg_cnt,kpt_avg_diff]



if __name__ == '__main__':

    model_path = 'weights_arcl2_cspdense_Rocket/thetabeta_ap_66_684_0.363_28.287.tar'    
    embede_model,_ = model_choose(model_path)

    save_path='best_results_diffsize_rocket/{}'.format(model_path.split('/')[-2])
    # fp_detector = cv2.xfeatures2d.SIFT_create(6000)
    # fp_detector = cv2.xfeatures2d.HarrisLaplaceFeatureDetector_create()
    fp_detector = cv2.FastFeatureDetector_create()
    #fp_detector = cv2.BRISK.create()
    start = time.time()

    data_path = 'test_image'
    mc,me,mex,mey,kpt_info = validate(embede_model,fp_detector,data_path,save_path)
    # print('{} test finished!'.format(model_path.split('/')[0].split('_')[1]))
    save_txt = '{}.txt'.format(model_path.split('/')[-2])
    save_data1 = 'matched key point average count: {}, key point average diff:{}\n'.format(kpt_info[0],kpt_info[1])
    save_data2 = 'match_count:%d, location match error:%f, x_error:%f, y_error:%f'%(mc,me,mex,mey)
    print('time consumed: %.2fs'%(time.time()-start))
    print(save_data1)
    print(save_data2)
    
        
