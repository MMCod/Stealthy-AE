
import os
import sys

lujing='xxx'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch,argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from models import ImagenetModels,osn_model
from rattacks import rfgsm
from rattacks import rpgd
from rattacks import rmifgsm
from rattacks import rcw
#from rattacks import helper
from utils import jpeg_qtableinv

def AttackLinf(imgs,labels,osnmodel,recogmodel,attype,eps,alpha=0.5,random=False,filter=False,steps=None
,cwc=None,lr=None,cwk=None,atttype=0):
    advs = None
    adv_p = labels
    flag = 0
    succ = False
    if random == True:
        attype = np.random.randint(4,size=1)[0]
    if attype == 0:
        e = int(eps)/255
        attack1 = rfgsm.RFGSM(recogmodel,osnmodel,eps=e,alpha=alpha)
        advs = attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        #print("type={},e={},l2={},l1={},alpha={}".format('UNIFGSM',e,l2,l1,alpha,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True  
    if attype==1 :
        eps = int(eps)/255
        attack1 = rmifgsm.RMIFGSM(recogmodel,osnmodel,eps=eps,alpha=alpha)
        advs= attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        #print("type={},e={},l2={},l1={},alpha={}".format('MIFGSM',eps,l2,l1,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)
        #print("target lable:{};predict label:{}".format(target_labels,adv_p))
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if attype==2:
        eps = int(eps)/255
        attack1 = rpgd.RPGD(recogmodel,osnmodel,eps=eps,balance=alpha)
        advs= attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        #print("type={},e={},l2={},l1={},alpha={}".format('UNIPGD',eps,l2,l1,alpha))
        _,adv_p = torch.max(osnmodel(advs),1)
        #print("target lable:{};predict label:{}".format(target_labels,adv_p))
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if attype == 3:
        c = cwc
        l2 =1000 
        st = steps
        llr = lr
        k =cwk
        attack1 = rcw.RCW(recogmodel,osnmodel,c=c,kappa=k,steps=st,lr=llr,balance=alpha,atttype=atttype) 
        advs = attack1(imgs,labels)
        l2 = torch.norm(imgs-advs)
        l1 = torch.norm(imgs-advs,1)
        #print("type={},c={},lr={},steps={},l2={},alpha={},k={}".format('UNICW',c,llr,st,l2,alpha,k))
        _,adv_p = torch.max(osnmodel(advs),1)   
        if torch.any(labels == adv_p):
            succ = False
        else:
            succ = True
    if filter == True:  
        if advs is None or torch.any(adv_p==labels):
            advs = 0  
    else:
        if advs is None:
            advs=0
    return advs, int(adv_p), attype, succ


