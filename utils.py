import os
import numpy as np
import torch
import shutil
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import itertools
import torch.nn as nn
from collections import defaultdict
import glob
import time
import imageio
import cifar_augmentations
import imagenet_augmentations
from autoaugment import ImageNetPolicy
from autoaugment import CIFAR10Policy
from collections import OrderedDict
import torch.nn.functional as F


class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


class AvgrageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape((-1)).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _robust_data_transforms_cifar10(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if args.cutout:
        transform_train.transforms.append(Cutout(args.cutout_length))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform_train, transform_test


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_cifar100(args):
    CIFAR_MEAN = [0.50705882, 0.48666667, 0.44078431]
    CIFAR_STD = [0.26745098, 0.25568627, 0.27607843]

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def _data_transforms_imagenet(args):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    if args.autoaugment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_size),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(args.train_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    valid_transform = transforms.Compose([
            transforms.Resize(args.eval_size + 32),
            transforms.CenterCrop(args.eval_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return train_transform, valid_transform


def count_parameters_woaux_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        print('Experiment dir : {}'.format(path))

        code_path = os.path.join(path, 'code')
        if not os.path.exists(code_path):
            os.makedirs(code_path)

        checkpoint_path = os.path.join(path, 'checkpoint')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        if scripts_to_save is not None:
            source_dir_path = os.path.abspath('./')
            for script in scripts_to_save:
                full_script_path = os.path.abspath(script)
                dst_path = os.path.join(code_path,
                                        full_script_path.replace(source_dir_path, '')[1:])
                if not os.path.exists(os.path.dirname(dst_path)):
                    os.makedirs(os.path.dirname(dst_path))
                shutil.copyfile(script, dst_path)
    except:
        print('')


def get_variable(inputs, device, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.tensor(inputs)
    out = Variable(inputs.to(device), **kwargs)
    return out


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


def aug(args, image, preprocess):
    """Perform AugMix augmentations and compute mixture.

    Args:
      image: PIL.Image input image
      preprocess: Preprocessing function which should return a torch tensor.

    Returns:
      mixed: Augmented and mixed image.
    """
    if 'cifar' in args.dataset:
        aug_list = cifar_augmentations.augmentations
        if args.all_ops:
            aug_list = cifar_augmentations.augmentations_all
    else:
        aug_list = imagenet_augmentations.augmentations
        if args.all_ops:
            aug_list = imagenet_augmentations.augmentations_all
    ws = np.float32(
        np.random.dirichlet([args.aug_prob_coeff] * args.mixture_width))
    m = np.float32(np.random.beta(args.aug_prob_coeff, args.aug_prob_coeff))

    mix = torch.zeros_like(preprocess(image))
    for i in range(args.mixture_width):
        image_aug = image.copy()
        depth = args.mixture_depth if args.mixture_depth > 0 else np.random.randint(
            1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, args.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""

    def __init__(self, args, dataset, preprocess):
        self.args = args
        self.dataset = dataset
        self.preprocess = preprocess

    def __getitem__(self, i):
        x, y = self.dataset[i]
        im_tuple = (self.preprocess(x), aug(self.args, x, self.preprocess),
                    aug(self.args, x, self.preprocess))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def diff_in_weights(model, proxy):
    EPS = 1E-20
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (new_k, new_w) in zip(model_state_dict.items(), proxy_state_dict.items()):
        if len(old_w.size()) <= 1:
            continue
        if 'weight' in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])


class AdvWeightPerturb(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(AdvWeightPerturb, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss = - F.cross_entropy(self.proxy(inputs_adv), targets)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class TradesAWP(object):
    def __init__(self, model, proxy, proxy_optim, gamma):
        super(TradesAWP, self).__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        loss_natural = F.cross_entropy(self.proxy(inputs_clean), targets)
        loss_robust = F.kl_div(F.log_softmax(self.proxy(inputs_adv), dim=1),
                               F.softmax(self.proxy(inputs_clean), dim=1),
                               reduction='batchmean')
        loss = - 1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


class Robust_KL_Loss(nn.KLDivLoss):
    def __init__(self, size_average=None, reduce=None, reduction='none', log_target=False):
        super(Robust_KL_Loss, self).__init__(size_average=size_average, reduce=reduce, reduction=reduction, log_target=log_target)

    def forward(self, input, target):
        batch_size = input.size(0)
        loss = F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)
        loss = (1.0 / batch_size) * torch.sum(torch.sum(loss, dim=1))
        return loss
