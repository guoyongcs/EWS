import argparse

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
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


parser = argparse.ArgumentParser("EWS")
parser.add_argument('--model_type', type=str, default='preresnet', help='resnet | preresnet | wideresnet')
parser.add_argument('--layers', type=int, default=18, help='total number of layers')
parser.add_argument('--test_only', action='store_true', default=False, help='test only')
# adversarial settings
parser.add_argument('--attack', default='pgd', type=str, choices=['pgd', 'fgsm', 'free', 'none'])
parser.add_argument('--epsilon', default=8, type=int)
parser.add_argument('--attack-iters', default=10, type=int)
parser.add_argument('--attack-iters-test', default=20, type=int)
parser.add_argument('--restarts', default=1, type=int)
parser.add_argument('--pgd-alpha', default=2, type=float)
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'])
parser.add_argument('--milestones', nargs='+', type=int, default=[100,150], help='milestones for learning rate')
parser.add_argument('--mixup', action='store_true')
parser.add_argument('--mixup_alpha', type=float)
# other settings
parser.add_argument('--resume', type=str, default='', help='resume from checkpoint')
parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
parser.add_argument('--data', type=str, default='./data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset: cifar10 | cifar100')
parser.add_argument('--train_interval_controller', type=int, default=10, help='training interval for controller')
parser.add_argument('--lambda_kl', type=float, default=0.1, help='weight of the KL loss')
parser.add_argument('--subnet_width', type=float, default=0.9, help='subnet width')
parser.add_argument('--drop_rate', type=float, default=0, help='drop rate')
parser.add_argument('--controller_hid', type=int, default=100, help='hidden dimension for controller')
parser.add_argument('--controller_temperature', type=float, default=None, help='temperature for lstm')
parser.add_argument('--controller_tanh_constant', type=float, default=None, help='tanh constant for lstm')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='coefficient for entropy')
parser.add_argument('--controller_learning_rate', type=float, default=3e-4, help='learning rate for controller')
parser.add_argument('--baseline_gamma', type=float, default=0.99, help='time decay for baseline update')
parser.add_argument('--lstm_num_layers', type=int, default=1, help='number of layers in lstm')
parser.add_argument('--controller_op_tanh_reduce', type=float, default=2.5, help='coefficient for entropy')
parser.add_argument('--num_workers', type=int, default=10, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--report_freq', type=int, default=50, help='report frequency')
parser.add_argument('--test_freq', type=int, default=20, help='test (go over all validset) frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--save', type=str, default='test', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=3, help='gradient clipping')
parser.add_argument('--ngpus', type=int, default=1, help='number of gpus')
parser.add_argument('--prefix', type=str, default='../experiments', help='parent save path')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--scheduler', type=str, default='step', help='type of LR scheduler')
parser.add_argument('--warmup_epochs', type=int, default=0, help='number of epochs for warmup')
parser.add_argument('--last_epoch', type=int, default=-1, help='last epoch to begin')
parser.add_argument('--awp-gamma', default=0.01, type=float)
parser.add_argument('--awp-warmup', default=0, type=int)
parser.add_argument('--awp', action='store_true', default=False, help='use awp')
parser.add_argument('--use_checkpoint', action='store_true', default=False, help='whether to use gradient checkpointing to save memory')

# load checkpoint
parser.add_argument('--checkpoint', type=str, default='', help='path for saved checkpoint')

args = parser.parse_args()

if args.dataset == 'cifar10':
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
elif args.dataset == 'cifar100':
    CIFAR_MEAN = [0.50705882, 0.48666667, 0.44078431]
    CIFAR_STD = [0.26745098, 0.25568627, 0.27607843]
mu = torch.tensor(CIFAR_MEAN).view(3, 1, 1).cuda()
std = torch.tensor(CIFAR_STD).view(3, 1, 1).cuda()

def normalize(X):
    return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)


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

    if args.dataset == 'cifar10':
        args.num_classes = 10
        train_transform, valid_transform = utils._robust_data_transforms_cifar10(args)
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    elif args.dataset == 'cifar100':
        args.num_classes = 100
        train_transform, valid_transform = utils._robust_data_transforms_cifar10(args)
        train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)
    else:
        assert False, 'not supported dataset %s' % args.dataset

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.num_workers)

    return train_queue, valid_queue


def set_model(args):
    if args.model_type not in ['preresnet', 'wideresnet']:
        assert False, 'model_type %s not supported' % args.model_type
    # cifar10 model
    if args.dataset == 'cifar10':
        if args.model_type == 'preresnet':
            model = PreActResNet_CIFAR(depth=args.layers, num_classes=10, subnet_width=args.subnet_width, drop_rate=args.drop_rate, use_checkpoint=args.use_checkpoint)
        elif args.model_type == 'wideresnet':
            model = WideResNet(depth=args.layers, subnet_width=args.subnet_width, use_checkpoint=args.use_checkpoint)
    # cifar100 model
    elif args.dataset == 'cifar100':
        if args.model_type == 'preresnet':
            model = PreActResNet_CIFAR(depth=args.layers, num_classes=100, subnet_width=args.subnet_width, drop_rate=args.drop_rate, use_checkpoint=args.use_checkpoint)
        elif args.model_type == 'wideresnet':
            model = WideResNet(depth=args.layers, num_classes=100, subnet_width=args.subnet_width, use_checkpoint=args.use_checkpoint)
    else:
        assert False, 'dataset %s not supported' % args.dataset
    return model


def set_scheduler(model, args):
    params = model.parameters()
    optimizer = torch.optim.SGD(params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.scheduler == 'step':
        logging.info(f'using milestones:{args.milestones}')
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=args.milestones, gamma=0.1, last_epoch=-1
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
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck, MaskBasicBlockCIFAR, MaskBottleneckCIFAR, MaskPreActBlock, MaskPreActBottleneck, MaskWideBasicBlock)):
            n_paths_config.append(module.n_paths)
            n_channels_config.append(module.n_channels)
            sub_paths_config.append(math.ceil(module.n_paths * args.subnet_width))
            sub_channels_config.append(
                [math.ceil(x * args.subnet_width) for x in module.n_channels]
            )
    return n_paths_config, n_channels_config, sub_paths_config, sub_channels_config


def switch_to_subnet(model, active_paths, active_indexes):
    deactivate_dropout(model)
    count_masklayer = 0
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck, MaskBasicBlockCIFAR, MaskBottleneckCIFAR, MaskPreActBlock, MaskPreActBottleneck, MaskWideBasicBlock)):
            module.sample_beta(active_paths[count_masklayer], active_indexes[count_masklayer])
            count_masklayer += 1
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0


def switch_to_fullnet(model):
    activate_dropout(model)
    for module in model.modules():
        if isinstance(module, (MaskBasicBlock, MaskBottleneck, MaskBasicBlockCIFAR, MaskBottleneckCIFAR, MaskPreActBlock, MaskPreActBottleneck, MaskWideBasicBlock)):
            module.reset_beta()
        if isinstance(module, nn.BatchNorm2d):
            module.momentum = 0.1


def update_baseline(baseline, reward):
    new_baseline = baseline * args.baseline_gamma + reward * (1-args.baseline_gamma)
    return new_baseline


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(normalize(X + delta))
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(normalize(X+delta)), y_a, y_b, lam)
        else:
            all_loss = F.cross_entropy(model(normalize(X+delta)), y, reduction='none')
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta


epsilon = (args.epsilon / 255.)
pgd_alpha = (args.pgd_alpha / 255.)

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
    proxy = set_model(args)
    logging.info("param size = %fMB", utils.count_parameters_woaux_in_MB(model))
    optimizer, scheduler = set_scheduler(model, args)
    proxy_opt = torch.optim.SGD(proxy.parameters(), lr=0.01)
    proxy_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        proxy_opt, milestones=args.milestones, gamma=0.1, last_epoch=-1
    )

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
    proxy = utils.dataparallel(proxy, args.ngpus)
    awp_adversary = utils.AdvWeightPerturb(model=model, proxy=proxy, proxy_optim=proxy_opt, gamma=args.awp_gamma)
    controller.to(device)
    if args.label_smooth > 0:
        criterion = utils.CrossEntropyLabelSmooth(args.num_classes, args.label_smooth)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    criterion_kl = utils.Robust_KL_Loss()
    criterion_kl = criterion_kl.to(device)

    if args.test_only:
        valid_acc_top1, valid_acc_top5, valid_adv_acc_top1, valid_adv_acc_top5, valid_obj = infer(valid_queue, model, criterion, device)
        logging.info(f'acc: {valid_acc_top1:.2f}, adv_acc: {valid_adv_acc_top1:.2f}')
        return

    for i in range(args.last_epoch - args.warmup_epochs):
        scheduler.step()
        proxy_scheduler.step()

    baseline = 0
    for epoch in range(args.last_epoch if args.last_epoch > -1 else 0, args.warmup_epochs + args.epochs):
        if epoch < args.warmup_epochs:
            warmup_update_lr(optimizer, epoch, args.learning_rate, args.warmup_epochs)
        else:
            scheduler.step()
            proxy_scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])

        train_acc, train_obj, model_remaining_time = train(train_queue, model, awp_adversary, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer, baseline)

        valid_acc_top1, valid_acc_top5, valid_adv_acc_top1, valid_adv_acc_top5, valid_obj = infer(valid_queue, model, criterion, device)

        is_best = False
        if valid_adv_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_adv_acc_top1
            best_acc_top5 = valid_adv_acc_top5
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
        logging.info(f'valid_acc: {valid_acc_top1:.2f}, adv_top1: {valid_adv_acc_top1:.2f}, best_adv_top1: {best_acc_top1:.2f}, best_adv_top5: {best_acc_top5:.2f}')

    logging.info('best_adv_acc_top1: %f, best_adv_acc_top5: %f at Epoch %d' % (best_acc_top1, best_acc_top5, best_epoch))


def train(train_queue, model, awp_adversary, criterion, criterion_kl, optimizer, device, epoch, controller, sub_paths_config, sub_channels_config, controller_optimizer=None, baseline=0):
    objs = utils.AvgrageMeter()
    adv_top1 = utils.AvgrageMeter()
    adv_top5 = utils.AvgrageMeter()
    log_epsilon=1e-12
    model.train()

    for step, (input, target) in enumerate(train_queue):
        n = input.size(0)
        input = input.to(device)
        target = target.to(device)

        switch_to_fullnet(model)
        # generate x_adv
        delta = attack_pgd(model, input, target, epsilon, pgd_alpha, args.attack_iters, args.restarts, args.norm)
        delta = delta.detach()
        x_adv = normalize(torch.clamp(input + delta[:input.size(0)], min=lower_limit, max=upper_limit))

        # train the controller
        if step % args.train_interval_controller== 0:
            controller_optimizer.zero_grad()
            active_paths, active_indexes, log_p, entropy = controller(sub_paths_config, sub_channels_config)
            # switch to subnet
            switch_to_subnet(model, active_paths, active_indexes)
            adv_student_logits = model(x_adv)
            accumulated_entropy = 0
            accumulated_logp = 0
            prec1 = utils.accuracy(adv_student_logits, target)[0]
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
        if args.awp:
            # calculate adversarial weight perturbation and perturb it
            awp = awp_adversary.calc_awp(inputs_adv=x_adv, targets=target)
            awp_adversary.perturb(awp)
        else:
            awp = None
        adv_logits = model(x_adv)
        adv_loss = criterion(adv_logits, target)
        teacher_adv_logits = adv_logits.detach()
        # switch to subnet
        switch_to_subnet(model, active_paths, active_indexes)
        adv_student_logits = model(x_adv)
        adv_student_probs = F.softmax(adv_student_logits, dim=1)
        adv_kl_loss = criterion_kl(torch.log(adv_student_probs + log_epsilon), F.softmax(teacher_adv_logits, dim=1))
        loss = adv_loss + args.lambda_kl * adv_kl_loss
        loss.backward()
        optimizer.step()

        if args.awp and epoch >= args.awp_warmup:
            awp_adversary.restore(awp)

        adv_prec1, adv_prec5 = utils.accuracy(adv_logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        adv_top1.update(adv_prec1.item(), n)
        adv_top5.update(adv_prec5.item(), n)

        if step % args.report_freq == 0:
            logging.info(f'train {epoch:03d} [{step:03d}/{len(train_queue):03d}], loss: {objs.avg:.2e}, top1: {adv_top1.avg:.2f}, top5: {adv_top5.avg:.2f}')

    return adv_top1.avg, objs.avg, baseline


def infer(valid_queue, model, criterion, device):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    adv_top1 = utils.AvgrageMeter()
    adv_top5 = utils.AvgrageMeter()

    switch_to_fullnet(model)
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.to(device)
        target = target.to(device)

        delta = attack_pgd(model, input, target, epsilon, pgd_alpha, args.attack_iters_test, 1, args.norm)
        delta = delta.detach()
        x_adv = normalize(torch.clamp(input + delta[:input.size(0)], min=lower_limit, max=upper_limit))
        x = normalize(input)

        logits = model(x)
        adv_logits = model(x_adv)
        loss = criterion(logits, target)
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        adv_prec1, adv_prec5 = utils.accuracy(adv_logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        adv_top1.update(adv_prec1.item(), n)
        adv_top5.update(adv_prec5.item(), n)

        if step % args.test_freq == 0:
            logging.info(f'valid {step:03d}, loss: {objs.avg:.2e}, top1: {top1.avg:.2f}, top5: {top5.avg:.2f}, adv_top1: {adv_top1.avg:.2f}, adv_top5: {adv_top5.avg:.2f}')

    return top1.avg, top5.avg, adv_top1.avg, adv_top5.avg, objs.avg


if __name__ == '__main__':
    main()
