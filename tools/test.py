from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch
import torch.nn.functional as F
import numpy as np
import pdb, os, argparse
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='6'
from scipy import misc
from scipy import ndimage
import imageio
from skimage import img_as_ubyte

from model import load_model
from data_process import test_dataset
import cv2
import sys

sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/data/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/lib/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/config/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/tools/')


parser = argparse.ArgumentParser()
#parser.add_argument('--testsize', type=int, default=[768,1536], help='testing size')
parser.add_argument('--testsize', type=int, default=[650,1300], help='testing size')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
opt = parser.parse_args()

dataset_path1 = '/157Dataset/data-chen.dongwen/icme_2017/original_data/Eval/Images/'
dataset_path2 = '/home/lab-chen.dongwen/ACSalNet/data/icme17_salicon_like/global_45/images/test/'


if opt.is_ResNet:
    model = load_model(classes=256,
                     node_size=(64, 128)).cuda()   

    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/best_testing_result_epoch99.pth'))   
else:
    model = CPD_VGG()
    model.load_state_dict(torch.load('/home/lab-chen.dongwen/saliency/mine/mymodel2_panorama/models/panorama/CPD_VGG/train_30_val_10/repetition_contextual_encoder-decoder_network_for_visual_saliency_prediction.pth'))

model.eval()

test_datasets = ['predictions_salmap'] 

for dataset in test_datasets:
    if opt.is_ResNet:
        save_path = '/157Dataset/data-chen.dongwen/icme_2017/graduation_project/' + dataset + '/'                         
    else:
        save_path = '/157Dataset/data-chen.dongwen/icme_2017/graduation_project/' + dataset + '/'                         

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root1 = dataset_path1
    image_root2 = dataset_path2             
    gt_root = dataset_path1             

    test_loader = test_dataset(image_root1, image_root2, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image1, image2, gt, name1, name2 = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image1 = image1.cuda()
        image2 = image2.cuda()

        res1, res2 = model(image1, image2)

        res1 = F.interpolate(res1, size=gt.shape, mode='bilinear', align_corners=False)  
        res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8) 

        res2 = F.interpolate(res2, size=gt.shape, mode='bilinear', align_corners=False)  
        res2 = res2.sigmoid().data.cpu().numpy().squeeze()
        res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8) 


        imageio.imwrite(save_path+name1, img_as_ubyte(res1))
        imageio.imwrite(save_path+name2, img_as_ubyte(res2))
        print(i)

print('Done')
