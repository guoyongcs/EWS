import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import os
import time
import glob
import utils
import logging
import sys
import numpy as np
import random
import torchvision.datasets as dset
from model import *
import datetime
import torch.nn.functional as F
import math
from search import *
from autoaugment import ImageNetPolicy

parser = argparse.ArgumentParser("EWS")


# Common settings
parser.add_argument('--test_only', action='store_true', default=False, help='test only')
parser.add_argument('--data', type=str, default='./data', help='location of the data')
parser.add_argument('--inc_path', type=str, default='./data', help='location of ImageNet-C')
parser.add_argument('--dataset', type=str, default='imagenet', help='dataset')
parser.add_argument('--train_size', type=int, default=224, help='train image size')
parser.add_argument('--eval_size', type=int, default=224, help='eval image size')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of epochs for warmup')
parser.add_argument('--no_bias_decay', action='store_true', default=False, help='no bias decay')
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=20, help='test (go over all validset) frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=180, help='number of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=3, help='gradient clipping')
parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
parser.add_argument('--last_epoch', type=int, default=-1, help='last epoch to begin')
parser.add_argument('--prefix', type=str, default='../experiments', help='parent save path')
parser.add_argument('--scheduler', type=str, default='step', help='type of LR scheduler: step | cosine')
# DeepAugment
parser.add_argument('--deepaugment', action='store_true', default=False, help='use deepaugment')
parser.add_argument('--deepaugment_base_path', type=str, default='/path/to/DeepAugment', help='path to deepaugment data')
# AugMix options
parser.add_argument('--augmix', action='store_true', default=False, help='use augmix')
parser.add_argument('--autoaugment', action='store_true', default=False, help='use autoaugment')
parser.add_argument('--mixture-width', default=3, type=int, help='Number of augmentation chains to mix per augmented example')
parser.add_argument('--mixture-depth', default=-1, type=int, help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
parser.add_argument('--aug-severity', default=1, type=int, help='Severity of base augmentation operators| 1 for imagenet and 3 for cifar')
parser.add_argument('--aug-prob-coeff', default=1., type=float, help='Probability distribution coefficients')
parser.add_argument('--all-ops', '-all', action='store_true', help='Turn on all operations (+brightness,contrast,color,sharpness).')
# params for model
parser.add_argument('--model_type', type=str, default='resnet', help='model type: resnet')
parser.add_argument('--layers', type=int, default=50, help='total number of layers')
parser.add_argument('--drop_rate', type=float, default=0, help='drop rate')
parser.add_argument('--lambda_kl', type=float, default=1.0, help='weight of the KL loss')
parser.add_argument('--subnet_width', type=float, default=0.7, help='subnet width')
parser.add_argument('--train_interval_controller', type=int, default=10, help='every epoch for training controller')
parser.add_argument('--controller_hid', type=int, default=100, help='hidden dimension for controller')
parser.add_argument('--controller_temperature', type=float, default=None, help='temperature for lstm')
parser.add_argument('--controller_tanh_constant', type=float, default=None, help='tanh constant for lstm')
parser.add_argument('--entropy_coeff', type=float, default=0.005, help='coefficient for entropy')
parser.add_argument('--controller_learning_rate', type=float, default=3e-4, help='learning rate for controller')
parser.add_argument('--baseline_gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='coefficient for entropy')
parser.add_argument('--use_checkpoint', action='store_true', default=False, help='whether to use gradient checkpointing to save memory')

# load checkpoint
parser.add_argument('--checkpoint', type=str, default='', help='path for saved checkpoint')

args = parser.parse_args()


args.save = '{}-{}'.format(parser.prog, args.save)
args.save = os.path.join(args.prefix, args.save)

utils.create_exp_dir(args.save, scripts_to_save=glob.glob('./**/*.py', recursive=True) + glob.glob('./**/*.sh', recursive=True) + glob.glob('./**/*.yml', recursive=True))
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('experiment directory created')


def set_dataloader(args):
    traindir = os.path.join(args.data, 'train')
    validdir = os.path.join(args.data, 'val')
    train_transform, valid_transform = utils._data_transforms_imagenet(args)

    if args.augmix:
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose(
            [transforms.RandomResizedCrop(224),
             transforms.RandomHorizontalFlip()])
        if args.autoaugment:
            train_transform.transforms.append(ImageNetPolicy())
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        if args.cutout:
            preprocess.transforms.append(utils.Cutout(args.cutout_length))
        train_data = dset.ImageFolder(traindir, train_transform)
        if args.deepaugment:
            edsr_data = dset.ImageFolder(os.path.join(args.deepaugment_base_path, 'EDSR'), train_transform)
            cae_data = dset.ImageFolder(os.path.join(args.deepaugment_base_path, 'CAE'), train_transform)
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
        train_data = utils.AugMixDataset(args, train_data, preprocess)
    else:
        train_data = dset.ImageFolder(traindir, train_transform)
        if args.deepaugment:
            edsr_data = dset.ImageFolder(os.path.join(args.deepaugment_base_path, 'EDSR'), train_transform)
            cae_data = dset.ImageFolder(os.path.join(args.deepaugment_base_path, 'CAE'), train_transform)
            train_data = torch.utils.data.ConcatDataset([train_data, edsr_data, cae_data])
    valid_data = dset.ImageFolder(validdir, valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return train_queue, valid_queue


def set_model(args):
    if args.model_type not in ['resnet']:
        assert False, 'model_type %s not supported' % args.model_type
    if args.dataset == 'imagenet':
        if args.model_type == 'resnet':
            model = ResNet_IMAGENET(depth=args.layers, num_classes=1000, subnet_width=args.subnet_width, drop_rate=args.drop_rate, use_checkpoint=args.use_checkpoint)
    else:
        assert False, 'dataset %s not supported' % args.dataset
    return model


def set_scheduler(model, args):
    group_weight = []
    group_bias = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            group_bias.append(param)
        else:
            group_weight.append(param)
    assert len(list(model.parameters())) == len(group_weight) + len(group_bias)
    optimizer = torch.optim.SGD([
        {'params': group_weight},
        {'params': group_bias, 'weight_decay': 0 if args.no_bias_decay else args.weight_decay}
    ], lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0.0001, last_epoch=-1
        )
    elif args.scheduler == 'step':
        '''
        lr = 0.1     if epoch < 1/3 * epochs
        lr = 0.01    if 1/3 * epochs <= epoch < 2/3 * epochs
        lr = 0.001   if epoch >= 2/3 * epochs
        '''
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[args.epochs//3, args.epochs//3*2], gamma=0.1, last_epoch=-1
        )
    else:
        assert False, "unsupported schudeler type: %s" % args.scheduler
    return optimizer, scheduler


def warmup_update_lr(optimizer, epoch, init_lr, warmup_epochs):
    """
    update learning rate of optimizers
    """
    lr = init_lr * (epoch+1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def activate_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = args.drop_rate

def deactivate_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0


def get_configurations(model):
    n_paths_config = []
    n_channels_config = []
    sub_paths_config = []
    sub_channels_config = []
    # add epsilon to avoid width=0
    subnet_width = args.subnet_width + 1e-6
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck, MaskBasicBlockCIFAR, MaskBottleneckCIFAR)):
            n_paths_config.append(module.n_paths)
            n_channels_config.append(module.n_channels)
            sub_paths_config.append(math.ceil(module.n_paths * subnet_width))
            sub_channels_config.append(
                [math.ceil(x * subnet_width) for x in module.n_channels]
            )
    return n_paths_config, n_channels_config, sub_paths_config, sub_channels_config


def switch_to_subnet(model, active_paths, active_indexes):
    deactivate_dropout(model)
    count_masklayer = 0
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck)):
            module.sample_beta(active_paths[count_masklayer], active_indexes[count_masklayer])
            count_masklayer += 1
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0


def switch_to_fullnet(model):
    activate_dropout(model)
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck)):
            module.reset_beta()
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1


def update_baseline(baseline, reward):
    new_baseline = baseline * args.baseline_gamma + reward * (1-args.baseline_gamma)
    return new_baseline


def main():
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info('GPU device = %d' % args.gpu)
    else:
        logging.info('no GPU available, use CPU!!')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    logging.info("args = %s", args)

    train_queue, valid_queue = set_dataloader(args)
    model = set_model(args)
    logging.info("param size = %fMB", utils.count_parameters_woaux_in_MB(model))
    optimizer, scheduler = set_scheduler(model, args)

    n_paths_config, n_channels_config, sub_paths_config, sub_channels_config = get_configurations(model)
    controller = Controller(n_paths_config, n_channels_config, device, controller_hid=args.controller_hid, controller_temperature=args.controller_temperature, controller_tanh_constant=args.controller_tanh_constant, controller_op_tanh_reduce=args.controller_op_tanh_reduce)
    controller_optimizer = torch.optim.Adam(controller.parameters(),
                                            lr=args.controller_learning_rate, betas=(0.5, 0.999))

    # load checkpoint for test
    if args.checkpoint != '':
        logging.info('load checkpoint %s' % args.checkpoint)
        cpt = torch.load(args.checkpoint)
        model.load_state_dict(cpt['state_dict'])

    best_acc_top1 = 0
    best_acc_top5 = 0
    best_epoch = 0
    # automatically resume the training
    cpt_path = os.path.join(args.save, 'checkpoint/checkpoint.pth')
    if os.path.exists(cpt_path):
        args.resume = cpt_path
    if args.resume != '':
        logging.info('load checkpoint %s' % args.resume)
        cpt = torch.load(args.resume)
        if os.path.basename(args.save) in args.resume:
            args.last_epoch = cpt['epoch']
            best_acc_top1 = cpt['best_acc_top1']
        else:
            args.last_epoch = cpt['epoch'] if args.last_epoch == -1 else args.last_epoch
        model.load_state_dict(cpt['state_dict'])
        controller.load_state_dict(cpt['controller_state_dict'])
        try:
            best_acc_top5 = cpt['best_acc_top5']
            best_epoch = cpt['best_epoch']
        except:
            logging.info('cannot load best_acc_top5 and best_epoch')

    model = utils.dataparallel(model, args.ngpus)
    controller.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_kl = nn.KLDivLoss()
    criterion_kl = criterion_kl.to(device)

    if args.test_only:
        clean_top1, clean_top5, _ = infer(valid_queue, model, criterion, device)
        logging.info(f'clean_top1: {clean_top1:.2f}, lean_top5: {clean_top5:.2f}')
        mce = eval_imagenetc(model, device)
        logging.info(f'mCE: {mce:.2f}')
        return

    for i in range(args.last_epoch - args.warmup_epochs):
        scheduler.step()

    baseline = 0
    for epoch in range(args.last_epoch if args.last_epoch > -1 else 0, args.warmup_epochs + args.epochs):
        if epoch < args.warmup_epochs:
            cur_lr = warmup_update_lr(optimizer, epoch, args.learning_rate, args.warmup_epochs)
        else:
            scheduler.step()
            cur_lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, cur_lr)

        if args.augmix:
            train_acc, train_obj, baseline = train_augmix(train_queue, model, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer, baseline)
        else:
            train_acc, train_obj, baseline = train(train_queue, model, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer, baseline)

        valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, device)

        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            best_acc_top5 = valid_acc_top5
            best_epoch = epoch + 1
            is_best = True

        try:
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'controller_state_dict': controller.module.state_dict() if isinstance(controller, nn.DataParallel) else controller.state_dict(),
                'best_acc_top1': best_acc_top1,
                'best_acc_top5': best_acc_top5,
                'best_epoch': best_epoch,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join(args.save, 'checkpoint'))
        except:
            logging.info('cannot save checkpoint')

        logging.info(f'epoch: {epoch}, valid_acc: {valid_acc_top1:.2f}, best_top1: {best_acc_top1:.2f}, best_top5: {best_acc_top5:.2f}')

    logging.info('best_acc_top1: %f, best_acc_top5: %f at Epoch %d' % (best_acc_top1, best_acc_top5, best_epoch))


def train(train_queue, model, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer=None, baseline=0):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)

        # train the controller
        if step % args.train_interval_controller== 0:
            controller_optimizer.zero_grad()
            active_paths, active_indexes, log_p, entropy = controller(sub_paths_config, sub_channels_config)
            # switch to subnet
            switch_to_subnet(model, active_paths, active_indexes)
            student_logits = model(input)
            accumulated_entropy = 0
            accumulated_logp = 0
            prec1 = utils.accuracy(student_logits, target)[0]
            accumulated_entropy += entropy
            accumulated_logp += log_p
            complementary_accuracy = -prec1.item()/100
            reward = complementary_accuracy - baseline if baseline else complementary_accuracy
            policy_loss = -accumulated_logp * reward - args.entropy_coeff * accumulated_entropy
            policy_loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip)
            controller_optimizer.step()
            baseline = update_baseline(baseline, reward)

        # train the model
        optimizer.zero_grad()
        active_paths, active_indexes, log_p, entropy = controller(sub_paths_config, sub_channels_config)
        # switch to full model
        switch_to_fullnet(model)
        logits = model(input)
        ce_loss = criterion(logits, target)
        # switch to subnet
        switch_to_subnet(model, active_paths, active_indexes)
        student_logits = model(input)
        teacher_logits = logits.detach()
        kl_loss = criterion_kl(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits, dim=1))
        loss = ce_loss + args.lambda_kl * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        switch_to_fullnet(model)

        if step % args.report_freq == 0:
            logging.info(f'train {epoch:03d} [{step:03d}/{len(train_queue):03d}], loss: {objs.avg:.2e}, top1: {top1.avg:.2f}, top5: {top5.avg:.2f}')

    return top1.avg, objs.avg, baseline


def train_augmix(train_queue, model, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer=None, baseline=0):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input[0].size(0)
        input = torch.cat(input, 0).to(device)
        target = target.to(device)

        # train the controller
        if step % args.train_interval_controller== 0:
            controller_optimizer.zero_grad()
            active_paths, active_indexes, log_p, entropy = controller(sub_paths_config, sub_channels_config)
            # switch to subnet
            switch_to_subnet(model, active_paths, active_indexes)
            student_logits_all = model(input)
            student_logits, student_logits_aug1, student_logits_aug2 = torch.split(student_logits_all, n)
            accumulated_entropy = 0
            accumulated_logp = 0
            prec1 = utils.accuracy(student_logits, target)[0]
            accumulated_entropy += entropy
            accumulated_logp += log_p
            complementary_accuracy = -prec1.item()/100
            reward = complementary_accuracy - baseline if baseline else complementary_accuracy
            policy_loss = -accumulated_logp * reward - args.entropy_coeff * accumulated_entropy
            policy_loss.backward()
            nn.utils.clip_grad_norm_(controller.parameters(), args.grad_clip)
            controller_optimizer.step()
            baseline = update_baseline(baseline, reward)

        # train the model
        optimizer.zero_grad()
        active_paths, active_indexes, log_p, entropy = controller(sub_paths_config, sub_channels_config)
        # switch to full model
        switch_to_fullnet(model)
        logits_all = model(input)
        logits, logits_aug1, logits_aug2 = torch.split(logits_all, n)
        ce_loss = criterion(logits, target)
        p_clean, p_aug1, p_aug2 = F.softmax(
            logits, dim=1), F.softmax(
            logits_aug1, dim=1), F.softmax(
            logits_aug2, dim=1)
        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        ce_loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                         F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.
        # switch to subnet
        switch_to_subnet(model, active_paths, active_indexes)
        student_logits_all = model(input)
        student_logits, student_logits_aug1, student_logits_aug2 = torch.split(student_logits_all, n)
        teacher_logits_clean = logits.detach()
        teacher_logits_aug1 = logits_aug1.detach()
        teacher_logits_aug2 = logits_aug2.detach()
        kl_loss = (criterion_kl(F.log_softmax(student_logits, dim=1), F.softmax(teacher_logits_clean, dim=1)) + criterion_kl(F.log_softmax(student_logits_aug1, dim=1), F.softmax(teacher_logits_aug1, dim=1)) + criterion_kl(F.log_softmax(student_logits_aug2, dim=1), F.softmax(teacher_logits_aug2, dim=1))) / 3.
        loss = ce_loss + args.lambda_kl * kl_loss
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        switch_to_fullnet(model)

        if step % args.report_freq == 0:
            logging.info(f'train {epoch:03d} [{step:03d}/{len(train_queue):03d}], loss: {objs.avg:.2e}, top1: {top1.avg:.2f}, top5: {top5.avg:.2f}')

    return top1.avg, objs.avg, baseline


def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()

    switch_to_fullnet(model)
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)
        logits = model(input)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        if step % args.test_freq == 0:
            logging.info(f'valid {step:03d}, loss: {objs.avg:.2e}, top1: {top1.avg:.2f}, top5: {top5.avg:.2f}')

    return top1.avg, top5.avg, objs.avg


def show_performance(distortion_name, model, device):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    errs = []

    INC_DIR = args.inc_path

    for severity in range(1, 6):
        print(distortion_name, severity)
        distorted_dataset = dset.ImageFolder(
            root=INC_DIR + '/' + distortion_name + '/' + str(severity),
            transform=transforms.Compose([transforms.Resize(args.eval_size + 32), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean, std)]))

        distorted_dataset_loader = torch.utils.data.DataLoader(
            distorted_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        correct = 0
        for batch_idx, (data, target) in enumerate(distorted_dataset_loader):
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.cuda()).sum()

            if batch_idx % args.test_freq == 0:
                logging.info(f'distortion_name: {distortion_name}, severity: {severity}, [{batch_idx}/{len(distorted_dataset_loader)}]')

        errs.append(1 - 1.*correct.item() / len(distorted_dataset))

    logging.info(f'Average: {errs}')
    return np.mean(errs)


def eval_imagenetc(model, device):
    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]

    ALEXNET_ERR = [
        0.886428, 0.894468, 0.922640, 0.819880, 0.826268, 0.785948, 0.798360,
        0.866816, 0.826572, 0.819324, 0.564592, 0.853204, 0.646056, 0.717840,
        0.606500
    ]

    error_rates = []
    mce = 0.
    i=0
    for distortion_name in distortions:
        rate = show_performance(distortion_name, model, device)
        error_rates.append(rate)
        logging.info('Distortion: {:15s}  | CE (unnormalized) (%): {:.2f}'.format(distortion_name, 100 * rate))
        ce = 100 * rate / ALEXNET_ERR[i]
        mce += ce / 15
        i += 1

    logging.info('mCE (unnormalized by AlexNet errors) (%): {:.2f}, mCE: {:.2f}'.format(100 * np.mean(error_rates), mce))
    return mce


if __name__ == '__main__':
    main()
