from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time
import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'

sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/data/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/lib/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/')
sys.path.append('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/config/')

import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

from config.configs_kf import *
from lib.utils.lookahead import *
from lib.utils.lr import init_params_lr
from lib.utils.measure import *
from model import load_model
from data_process import get_loader, get_loader_val
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
cudnn.benchmark = True


################## parameters in ICME2017 image dataset #########################3
KLmiu = 2.4948
KLstd = 1.7421
CCmiu = 0.3932
CCstd = 0.2565
NSSmiu = 0.4539
NSSstd = 0.2631
bcemiu = 0.3194
bcestd = 0.1209
alpha = 0.05
###############################################################

def KL_div(output, target):
    output = output / torch.sum(output)
    target = target / torch.sum(target)
    a = target * torch.log(target / (output+1e-20)+1e-20)
    b = output * torch.log(output / (target+1e-20)+1e-20)
    return (torch.sum(a) + torch.sum(b)) / (2.)

def CC(output, target):
    output = (output - torch.mean(output)) / torch.std(output)
    target = (target - torch.mean(target)) / torch.std(target)
    num = (output - torch.mean(output)) * (target - torch.mean(target))
    out_square = (output - torch.mean(output)) * (output - torch.mean(output))
    tar_square = (target - torch.mean(target)) * (target - torch.mean(target))
    CC_score = torch.sum(num) / (torch.sqrt(torch.sum(out_square) * torch.sum(tar_square)))
    return CC_score

def NSS(output, fixationMap):
    output = (output-torch.mean(output))/torch.std(output)
    Sal = output*fixationMap
    NSS_score = torch.sum(Sal)/torch.sum(fixationMap)
    return NSS_score



train_args = agriculture_configs(net_name='MSCG-Rx50',
                                 data='Agriculture',
                                 bands_list=['NIR', 'RGB'],
                                 kf=0, k_folder=0,
                                 note='reproduce_ACW_loss2_adax'
                                 )

train_args.input_size = [512, 512]
train_args.scale_rate = 1.  # 256./512.  # 448.0/512.0 #1.0/1.0
train_args.val_size = [512, 512]
train_args.node_size = (64, 128)
train_args.train_batch = 6
train_args.val_batch = 6

train_args.lr = 1.5e-4 / np.sqrt(3)
train_args.weight_decay = 2e-5

train_args.lr_decay = 0.9
train_args.max_iter = 1e8

train_args.snapshot = ''

train_args.print_freq = 100
train_args.save_pred = False
train_args.save_path = '/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/ckpt/'
# output training configuration to a text file
train_args.ckpt_path=os.path.abspath(os.curdir)

writer = SummaryWriter(os.path.join(train_args.save_path, 'tblog'))

image_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_image256x192/'
gt_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_salmap256x192/'
fixmap_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_fixmap256x192/'
image_val_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_image256x192/'
gt_val_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_salmap256x192/'
fixmap_val_root = '/157Dataset/data-chen.dongwen/icme_2017/suitable_preprocessingdata_m/original_fixmap256x192/'



def validation_sample_visualization(path1, path2, gt_path, model, epoch):
    transform = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_gt = transforms.Compose([
        transforms.Resize((512, 1024)),
        transforms.ToTensor()])

    validation_sample1 = Image.open(path1) 
    validation_sample1.convert('RGB')
    validation_sample1 = transform(validation_sample1).unsqueeze(0)
    validation_sample1 = validation_sample1.cuda()

    validation_sample2 = Image.open(path2) 
    validation_sample2.convert('RGB')
    validation_sample2 = transform(validation_sample2).unsqueeze(0)
    validation_sample2 = validation_sample2.cuda()

    validation_gt = Image.open(gt_path) 
    validation_gt.convert('RGB')
    validation_gt = transform_gt(validation_gt).unsqueeze(0)
    validation_gt = validation_gt.cuda()

    outputs1, outputs2 = model(validation_sample1, validation_sample2)

    outputs1 = F.upsample(outputs1, size=(512, 1024), mode='bilinear', align_corners=False)
    outputs1 = outputs1.sigmoid()
    outputs1 = (outputs1 - outputs1.min()) / (outputs1.max() - outputs1.min() + 1e-8)
    outputs1 = torch.cuda.FloatTensor(outputs1)
    outputs1 = vutils.make_grid(outputs1, normalize=True, scale_each=True)

    outputs2 = F.upsample(outputs2, size=(512, 1024), mode='bilinear', align_corners=False)
    outputs2 = outputs2.sigmoid()
    outputs2 = (outputs2 - outputs2.min()) / (outputs2.max() - outputs2.min() + 1e-8)
    outputs2 = torch.cuda.FloatTensor(outputs2)
    outputs2 = vutils.make_grid(outputs2, normalize=True, scale_each=True)

    validation_gt = (validation_gt - validation_gt.min()) / (validation_gt.max() - validation_gt.min() + 1e-8)
    validation_gt = torch.cuda.FloatTensor(validation_gt)
    validation_gt = vutils.make_grid(validation_gt, normalize=True, scale_each=True)
    writer.add_image('outputs1', outputs1, epoch)
    writer.add_image('outputs2', outputs2, epoch)
    writer.add_image('gt', validation_gt, epoch)


def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def main():
    random_seed(train_args.seeds)
    train_args.write2txt()
    net = load_model(name=train_args.model, classes=256,
                     node_size=train_args.node_size)   #what is nb_classes?

    net, start_epoch = train_args.resume_train(net)

    net = torch.nn.DataParallel(net)

    net.cuda()

    ### load the pretrained model
    model_dict = net.state_dict()
    pretrained_dict = torch.load('/home/lab-chen.dongwen/graduation_project/mine/mymodel3_panorama/best_finetune_result_epoch54.pth')
    pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

    net.train()

    train_loader = get_loader(image_root, gt_root, fixmap_root, batchsize=train_args.train_batch, trainsize=train_args.input_size)
    val_loader = get_loader_val(image_val_root, gt_val_root, fixmap_val_root, batchsize=train_args.train_batch, trainsize=train_args.input_size)

    criterion = torch.nn.BCEWithLogitsLoss()

    params = init_params_lr(net, train_args)

    new_ep = 0
    for epoch in range(1, 100):
        starttime = time.time()
        train_main_loss = AverageMeter()
        aux_train_loss = AverageMeter()
        cls_trian_loss = AverageMeter()

        if epoch < 10:
            base_optimizer = optim.Adam(params, amsgrad=True)
        else:
            base_optimizer = optim.SGD(params, momentum=train_args.momentum, nesterov=True)

        optimizer = Lookahead(base_optimizer, k=6)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, 1.18e-6)

        start_lr = train_args.lr
        train_args.lr = optimizer.param_groups[0]['lr']
        num_iter = len(train_loader)
        curr_iter = ((start_epoch + new_ep) - 1) * num_iter
        print('---curr_iter: {}, num_iter per epoch: {}---'.format(curr_iter, num_iter))

        for i, (inputs1, inputs2, labels1, labels2, fixmaps1, fixmaps2) in enumerate(train_loader):

            inputs1, inputs2, labels1, labels2, fixmaps1, fixmaps2 = inputs1.cuda(), inputs2.cuda(), labels1.cuda(), labels2.cuda(), fixmaps1.cuda(), fixmaps2.cuda()

            N = inputs1.size(0) * inputs1.size(2) * inputs1.size(3)
            optimizer.zero_grad()

            outputs1, outputs2 = net(inputs1, inputs2)

            main_loss1 = criterion(outputs1, labels1) + alpha * (bcemiu + bcestd * ((1.)*((CC(outputs1, labels1) - CCmiu) / CCstd) - (1.)*((NSS(outputs1, fixmaps1) - NSSmiu) / NSSstd)))
            main_loss2 = criterion(outputs2, labels2) + alpha * (bcemiu + bcestd * ((1.)*((CC(outputs2, labels2) - CCmiu) / CCstd) - (1.)*((NSS(outputs2, fixmaps2) - NSSmiu) / NSSstd)))
            loss = main_loss1 + main_loss2

            loss.mean().backward()
            optimizer.step()
            lr_scheduler.step(epoch=(start_epoch + new_ep))

            train_main_loss.update(loss.item(), N)

            curr_iter += 1
            writer.add_scalar('main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_iter)

            if (i + 1) % train_args.print_freq == 0:
                newtime = time.time()

                print('[epoch %d], [iter %d / %d], [loss %.5f, aux %.5f, cls %.5f], [lr %.10f], [time %.3f]' %
                      (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg, aux_train_loss.avg,
                       cls_trian_loss.avg,
                       optimizer.param_groups[0]['lr'], newtime - starttime))

                starttime = newtime

        validate(net, val_loader, criterion, optimizer, start_epoch + new_ep, new_ep)

        new_ep += 1


def validate(net, val_loader, criterion, optimizer, epoch, new_ep):
    net.eval()
    val_loss = AverageMeter()

    num_iter = len(val_loader)

    with torch.no_grad():
        for vi, (inputs1, inputs2, gts1, gts2, fixmaps1, fixmaps2) in enumerate(val_loader):

            inputs1, inputs2, gts1, gts2, fixmaps1, fixmaps2 = inputs1.cuda(), inputs2.cuda(), gts1.cuda(), gts2.cuda(), fixmaps1.cuda(), fixmaps2.cuda()

            N = inputs1.size(0) * inputs1.size(2) * inputs1.size(3)
            outputs1, outputs2 = net(inputs1, inputs2)

            main_loss1 = criterion(outputs1, gts1) + alpha * (bcemiu + bcestd * ((1.)*((CC(outputs1, gts1) - CCmiu) / CCstd) - (1.)*((NSS(outputs1, gts1) - NSSmiu) / NSSstd)))
            main_loss2 = criterion(outputs2, gts2) + alpha * (bcemiu + bcestd * ((1.)*((CC(outputs2, gts2) - CCmiu) / CCstd) - (1.)*((NSS(outputs2, fixmaps2) - NSSmiu) / NSSstd)))
            loss = main_loss1 + main_loss2
            val_loss.update(loss.item(), N)


            if (vi + 1) % train_args.print_freq == 0:

                print('[epoch %d], [iter %d / %d], [loss %.5f]' %
                      (epoch, vi + 1, num_iter, loss.item()))

    update_ckpt(net, optimizer, epoch, new_ep, val_loss)

    val_sample_path1 = '/home/lab-chen.dongwen/ACSalNet/data/icme17_salicon_like/global_45/images/test/P60_21.jpg'
    val_sample_path2 = '/home/lab-chen.dongwen/ACSalNet/data/icme17_salicon_like/global_45/images/test/P60_25.jpg'
    val_sample_gt_path = '/157Dataset/data-chen.dongwen/icme_2017/original_data/Eval/HeadEyeMaps/SHE60.jpg'
    validation_sample_visualization(val_sample_path1, val_sample_path2, val_sample_gt_path, net, epoch)

    net.train()
    return loss


def update_ckpt(net, optimizer, epoch, new_ep, val_loss):
    avg_loss = val_loss.avg

    writer.add_scalar('val_loss', avg_loss, epoch)

    snapshot_name = 'epoch_%d_loss_%.5f' % (
        epoch, avg_loss
    )

    if (epoch+1) % 5 == 0:
        torch.save(net.state_dict(), os.path.join(train_args.save_path, snapshot_name + '.pth'))


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


if __name__ == '__main__':
    main()
