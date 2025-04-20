
import os
import sys
import json

lujing='xxx'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch,argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from models import ImagenetModels, osn_model
from rattacks import rfgsm
from rattacks import rpgd
from rattacks import rmifgsm
from rattacks import rcw
#from rattacks import helper
from utils import jpeg_qtableinv
from rattack import *

classes = json.load(open(lujing+'classes.json','r+'))

transform = transforms.Compose([
    transforms.Resize(256),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


data_path=lujing+'imagenet/'

eps = 3 #linfty constratin for R-FGSM/R-PGD/R-MIFGSM
st = 100 #iteration step for R-C&W/R-PGD/R-MIFGSM
alpha = 0.5 #lambda in the paper for adjusting the loss function items
cwc = 1 #coeffecient c for R-C&W


attype = 2
#osn_name='facebook'
#base_model='resnet'

for i9 in ['vgg', 'incep']: #['resnet', 'incep', 'vgg']:
    base_model=i9
    for i8 in ['facebook', 'qq', 'wechat']:
        osn_name=i8
        chu=lujing+'attack/'+str(attype)+'='+osn_name+'='+base_model+'/'
        if os.path.exists(chu):
            continue
        os.mkdir(chu)
        print('base: ', base_model, 'osn: ', osn_name)
        if base_model=='resnet':
            recgmodel = ImagenetModels.Resnet50_Imagenet()
        elif base_model=='incep':
            recgmodel = ImagenetModels.IncepV3_Imagenet()
        elif base_model=='vgg':
            recgmodel = ImagenetModels.VGG19bn_Imagenet()
        osnmodel = osn_model.OSN_Model(target_model=recgmodel, osn_type=osn_name, base_path=lujing+'model/')
        fail = 0 
        suc = 0
        recog_suc = 0
        model_fail = 0
        model_suc = 0
        diff_advs, adv_labels, true_labels = [], [], []
        for i1 in range(1499):
            print('='*3, i1, '='*3)
            #if model_suc==200:
            #    break
            duqu = Image.open(data_path+str(i1)+'.png').convert('RGB')
            image_tensor = transform(duqu).unsqueeze(0)
            output = recgmodel(image_tensor)
            _, preds = torch.max(output, 1)
            #print('pred: ', preds.item(), 'true: ', classes[i1])
            if preds.item() != int(classes[i1]):
                model_fail += 1
                continue
            x = image_tensor.cuda()
            y = torch.tensor(int(classes[i1])).unsqueeze(0).cuda()
            #x_qf = int(jpeg_qtableinv(paths[0]))
            model_suc+=1
            lr = 0.01
            cwk=0
            x_adv, adv_label, attype, succ = AttackLinf(x, y, osnmodel, recgmodel, attype, eps, alpha=alpha, random=False,filter=False, steps=st, cwc=cwc, lr=lr, cwk=cwk, atttype=1)
            if x_adv is not None and x_adv.__class__.__name__ == 'Tensor':
                if succ:
                    succ_f = 1
                    suc += len(torch.where(adv_label!=y)[0])
                    #print("attack {} success".format(i1))
                else:
                    succ_f = 0
                    fail += 1
                    #print("attack {} fail".format(i1))
                _, adv_recog = torch.max(recgmodel(x_adv),1)
                if torch.all(adv_recog!=y):
                    vallina_suc =1
                    recog_suc +=1
                else:
                    vallina_suc =0
                adv_img = Image.fromarray(np.asarray(x_adv[0].cpu().permute(1,2,0).numpy()*255,dtype=np.uint8))
                imgpath = '/{}_true{}_adv{}_vall{}_osn{}.png'.format(i1,int(y.cpu()),adv_label,vallina_suc,succ_f)
                adv_img.save(chu+imgpath)
                diff_adv = torch.norm(x_adv-x)
                #print("advs l2 = {}".format(diff_adv))
                diff_advs.append(float(diff_adv.cpu()))            
                adv_labels.append(adv_label)
                true_labels.append(int(y.cpu()))
            else:
                #print("attack {} fail outer".format(i1))
                fail += 1 
        print("attack osnmodel success rate:{}".format(suc/model_suc))
        print("model success rate:{}".format(1-(model_fail/(model_suc+model_fail))))
        print("mean diff_advs ={}".format(np.mean(np.asarray(diff_advs))))
        print("attack vallina model success rate:{}".format(recog_suc/model_suc))



