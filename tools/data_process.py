import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, fixmap_root, trainsize):
        self.trainsize = trainsize
        self.images1 = [image_root + f for f in os.listdir(image_root) if f.startswith('train') and (f.endswith('_111.jpg') or f.endswith('_112.jpg') or f.endswith('_115.jpg') or f.endswith('_121.jpg') or f.endswith('_122.jpg') or f.endswith('_125.jpg') or f.endswith('_211.jpg') or f.endswith('_212.jpg') or f.endswith('_215.jpg') or f.endswith('_221.jpg') or f.endswith('_222.jpg') or f.endswith('_225.jpg'))]
        self.images2 = [image_root + f for f in os.listdir(image_root) if f.startswith('train') and (f.endswith('_113.jpg') or f.endswith('_114.jpg') or f.endswith('_116.jpg') or f.endswith('_123.jpg') or f.endswith('_124.jpg') or f.endswith('_126.jpg') or f.endswith('_213.jpg') or f.endswith('_214.jpg') or f.endswith('_216.jpg') or f.endswith('_223.jpg') or f.endswith('_224.jpg') or f.endswith('_226.jpg'))]
        self.gts1 = [gt_root + f for f in os.listdir(gt_root) if f.startswith('train') and (f.endswith('_111.png') or f.endswith('_112.png') or f.endswith('_115.png') or f.endswith('_121.png') or f.endswith('_122.png') or f.endswith('_125.png') or f.endswith('_211.png') or f.endswith('_212.png') or f.endswith('_215.png') or f.endswith('_221.png') or f.endswith('_222.png') or f.endswith('_225.png'))]
        self.gts2 = [gt_root + f for f in os.listdir(gt_root) if f.startswith('train') and (f.endswith('_113.png') or f.endswith('_114.png') or f.endswith('_116.png') or f.endswith('_123.png') or f.endswith('_124.png') or f.endswith('_126.png') or f.endswith('_213.png') or f.endswith('_214.png') or f.endswith('_216.png') or f.endswith('_223.png') or f.endswith('_224.png') or f.endswith('_226.png'))]
        self.fixmaps1 = [fixmap_root + f for f in os.listdir(fixmap_root) if f.startswith('train') and (f.endswith('_111.png') or f.endswith('_112.png') or f.endswith('_115.png') or f.endswith('_121.png') or f.endswith('_122.png') or f.endswith('_125.png') or f.endswith('_211.png') or f.endswith('_212.png') or f.endswith('_215.png') or f.endswith('_221.png') or f.endswith('_222.png') or f.endswith('_225.png'))]
        self.fixmaps2 = [fixmap_root + f for f in os.listdir(fixmap_root) if f.startswith('train') and (f.endswith('_113.png') or f.endswith('_114.png') or f.endswith('_116.png') or f.endswith('_123.png') or f.endswith('_124.png') or f.endswith('_126.png') or f.endswith('_213.png') or f.endswith('_214.png') or f.endswith('_216.png') or f.endswith('_223.png') or f.endswith('_224.png') or f.endswith('_226.png'))]
        self.images1 = sorted(self.images1)
        self.images2 = sorted(self.images2)
        self.gts1 = sorted(self.gts1)
        self.gts2 = sorted(self.gts2)
        self.fixmaps1 = sorted(self.fixmaps1)
        self.fixmaps2 = sorted(self.fixmaps2)


        self.size = len(self.images1)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize[0], self.trainsize[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize[0], self.trainsize[1])),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image1 = self.rgb_loader(self.images1[index])
        image2 = self.rgb_loader(self.images2[index])
        gt1 = self.binary_loader(self.gts1[index])
        gt2 = self.binary_loader(self.gts2[index])
        fixmap1 = self.binary_loader(self.fixmaps1[index])
        fixmap2 = self.binary_loader(self.fixmaps2[index])
        image1 = self.img_transform(image1)
        image2 = self.img_transform(image2)
        gt1 = self.gt_transform(gt1)
        gt2 = self.gt_transform(gt2)
        fixmap1 = self.gt_transform(fixmap1)
        fixmap2 = self.gt_transform(fixmap2)
        return image1, image2, gt1, gt2, fixmap1, fixmap2


    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size 
#        return self.size // 2

class SalObjDataset_val(data.Dataset):
    def __init__(self, image_val_root, gt_val_root, fixmap_val_root, trainsize):
        self.trainsize = trainsize

        self.images_val1 = [image_val_root + f for f in os.listdir(image_val_root) if f.startswith('train') and (f.endswith('_111.jpg') or f.endswith('_112.jpg') or f.endswith('_115.jpg') or f.endswith('_121.jpg') or f.endswith('_122.jpg') or f.endswith('_125.jpg') or f.endswith('_211.jpg') or f.endswith('_212.jpg') or f.endswith('_215.jpg') or f.endswith('_221.jpg') or f.endswith('_222.jpg') or f.endswith('_225.jpg'))]
        self.images_val2 = [image_val_root + f for f in os.listdir(image_val_root) if f.startswith('train') and (f.endswith('_113.jpg') or f.endswith('_114.jpg') or f.endswith('_116.jpg') or f.endswith('_123.jpg') or f.endswith('_124.jpg') or f.endswith('_126.jpg') or f.endswith('_213.jpg') or f.endswith('_214.jpg') or f.endswith('_216.jpg') or f.endswith('_223.jpg') or f.endswith('_224.jpg') or f.endswith('_226.jpg'))]
        self.gts_val1 = [gt_val_root + f for f in os.listdir(gt_val_root) if f.startswith('train') and (f.endswith('_111.png') or f.endswith('_112.png') or f.endswith('_115.png') or f.endswith('_121.png') or f.endswith('_122.png') or f.endswith('_125.png') or f.endswith('_211.png') or f.endswith('_212.png') or f.endswith('_215.png') or f.endswith('_221.png') or f.endswith('_222.png') or f.endswith('_225.png'))]
        self.gts_val2 = [gt_val_root + f for f in os.listdir(gt_val_root) if f.startswith('train') and (f.endswith('_113.png') or f.endswith('_114.png') or f.endswith('_116.png') or f.endswith('_123.png') or f.endswith('_124.png') or f.endswith('_126.png') or f.endswith('_213.png') or f.endswith('_214.png') or f.endswith('_216.png') or f.endswith('_223.png') or f.endswith('_224.png') or f.endswith('_226.png'))]
        self.fixmaps_val1 = [fixmap_val_root + f for f in os.listdir(fixmap_val_root) if f.startswith('train') and (f.endswith('_111.png') or f.endswith('_112.png') or f.endswith('_115.png') or f.endswith('_121.png') or f.endswith('_122.png') or f.endswith('_125.png') or f.endswith('_211.png') or f.endswith('_212.png') or f.endswith('_215.png') or f.endswith('_221.png') or f.endswith('_222.png') or f.endswith('_225.png'))]
        self.fixmaps_val2 = [fixmap_val_root + f for f in os.listdir(fixmap_val_root) if f.startswith('train') and (f.endswith('_113.png') or f.endswith('_114.png') or f.endswith('_116.png') or f.endswith('_123.png') or f.endswith('_124.png') or f.endswith('_126.png') or f.endswith('_213.png') or f.endswith('_214.png') or f.endswith('_216.png') or f.endswith('_223.png') or f.endswith('_224.png') or f.endswith('_226.png'))]
        self.images_val1 = sorted(self.images_val1)
        self.images_val2 = sorted(self.images_val2)
        self.gts_val1 = sorted(self.gts_val1)
        self.gts_val2 = sorted(self.gts_val2)
        self.fixmaps_val1 = sorted(self.fixmaps_val1)
        self.fixmaps_val2 = sorted(self.fixmaps_val2)


        self.size = len(self.images_val1)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize[0], self.trainsize[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize[0], self.trainsize[1])),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image_val1 = self.rgb_loader(self.images_val1[index])
        image_val2 = self.rgb_loader(self.images_val2[index])
        gt_val1 = self.binary_loader(self.gts_val1[index])
        gt_val2 = self.binary_loader(self.gts_val2[index])
        fixmap_val1 = self.binary_loader(self.fixmaps_val1[index])
        fixmap_val2 = self.binary_loader(self.fixmaps_val2[index])
        image_val1 = self.img_transform(image_val1)
        image_val2 = self.img_transform(image_val2)
        gt_val1 = self.gt_transform(gt_val1)
        gt_val2 = self.gt_transform(gt_val2)
        fixmap_val1 = self.gt_transform(fixmap_val1)
        fixmap_val2 = self.gt_transform(fixmap_val2)
        return image_val1, image_val2, gt_val1, gt_val2, fixmap_val1, fixmap_val2



    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


    def __len__(self):
        return self.size
#        return self.size // 2

def get_loader(image_root, gt_root, fixmap_root, batchsize, trainsize, shuffle=True, num_workers=12, pin_memory=True):

    dataset = SalObjDataset(image_root, gt_root, fixmap_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

def get_loader_val(image_val_root, gt_val_root, fixmap_val_root, batchsize, trainsize, shuffle=False, num_workers=12, pin_memory=True):

    dataset = SalObjDataset_val(image_val_root, gt_val_root, fixmap_val_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class test_dataset:
    def __init__(self, image_root1, image_root2, gt_root, testsize):
        self.testsize = testsize
        self.images1 = [image_root1 + f for f in os.listdir(image_root1) if f.endswith('.jpg')]
        self.images2 = [image_root2 + f for f in os.listdir(image_root2) if f.endswith('_25.jpg')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
#        self.images1 = sorted(self.images1)
#        self.images2 = sorted(self.images2)
        self.images = self.images1 + self.images2
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize[0], self.testsize[1])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images) // 2
        self.index = 0
        self.index_gt = 0

    def load_data(self):
        image1 = self.rgb_loader(self.images[self.index*2])
        image2 = self.rgb_loader(self.images[self.index*2+1])

        image1 = self.transform(image1).unsqueeze(0)
        image2 = self.transform(image2).unsqueeze(0)

        gt = self.binary_loader(self.gts[self.index_gt])
        if '_38' in self.images[self.index*2]:
            self.index_gt += 1

        name1 = self.images[self.index*2].split('/')[-1]
        name2 = self.images[self.index*2+1].split('/')[-1]
        if name1.endswith('.jpg'):
            name1 = name1.split('.jpg')[0] + '.png'
            name2 = name2.split('.jpg')[0] + '.png'
        self.index += 1
        return image1, image2, gt, name1, name2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


