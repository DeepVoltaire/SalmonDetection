import os, time
import numpy as np
import torchvision.models as models
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Sampler
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

classes = ["Net_present"]

def main(arch, model_name, train_loader, val_loader, epochs=1000, lr=0.001, patience=2, print_every_x_batches=1000):
    """
    Trains a classification network.

    :param arch, str: Architecture of pretrained backbone. "resnet50", "resnet34" and "resnet18" are possible.
    :param model_name, str: Name of the experiment.
    :param train_loader, torch.utils.data.Dataloader: Dataloader that yields training batches of images and targets.
    :param val_loader, torch.utils.data.Dataloader: Dataloader that yields validation batches of images and targets.
    :param epochs, int: How many epochs to train.
    :param lr, float: Learning rate for Adam optimizer to start with.
    :param patience, int: How many epochs without validation improvement until the learning rate is reduced and training
    is early stopped (patience times 3).
    :param print_every_x_batches, int: Print training information every how many training batches.

    """
    if arch=="resnet50": model = models.resnet50(pretrained=True)
    elif arch=="resnet34": model = models.resnet34(pretrained=True)
    elif arch=="resnet18": model = models.resnet18(pretrained=True)
    else: raise NotImplementedError()

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(classes), bias=True)
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, eps=1e-08)

    os.makedirs(f"trained_models/{model_name}", exist_ok=True)
    
    best_acc1, best_loss = 0, 999
    early_stop_counter, early_patience, early_stop_flag = 0, int(patience*3), False

    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, print_every_x_batches)

        # evaluate on validation set
        loss, acc1 = validate(val_loader, model, criterion)
        lr_scheduler.step(loss)

        # remember best loss
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        # remember best acc@1 and save checkpoint
        if acc1 > best_acc1: best_acc1_epoch = epoch + 1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'best_loss': best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best, model_name=model_name)

        # check if to early stop
        if not is_best:
            early_stop_counter += 1
            lr = optimizer.param_groups[0]['lr']
            if early_stop_counter > early_patience and lr < 1e-4:
                print("Early Stopping")
                early_stop_flag = True
        else:
            best_loss_epoch = epoch + 1
            early_stop_counter = 0
        if early_stop_flag: break
    print(f"Best loss of {best_loss:.4f} at epoch {best_loss_epoch}")
    print(f"Best Acc@1: {best_acc1:.4f} at epoch {best_acc1_epoch}")


def train(train_loader, model, criterion, optimizer, epoch, print_every_x_batches):
    """
    One training epoch.
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images, target = sample["image"], sample["target"]

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)[:, 0]
        loss = criterion(output, target.float())
        output = torch.sigmoid(output)

        # measure accuracy and record loss
        losses.update(loss.item(), images.size(0))
        acc1 = ((output > 0.5) == target.bool()).sum().float() / output.size(0)
        top1.update(acc1, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i > 0 and i % print_every_x_batches == 0:
            progress.display(i)
    progress.display(i)

def validate(val_loader, model, criterion):
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.4f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            
            images, target = sample["image"], sample["target"]
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)[:, 0]
            loss = criterion(output, target.float())
            output = torch.sigmoid(output)

            # measure accuracy and record loss
            losses.update(loss.item(), images.size(0))
            acc1 = ((output > 0.5) == target.bool()).sum().float() / output.size(0)
            top1.update(acc1, images.size(0))

        print(f'Val: Loss {losses.avg:.4f} Acc@1 {top1.avg:.4f}')

    return losses.avg, top1.avg

def save_checkpoint(state, is_best, model_name="test"):
    if is_best: torch.save(state, f"trained_models/{model_name}/model_best.pth.tar")


class SalmonDataset(Dataset):
    """
    Dataset for training a Classification network to detect Fish nets.
    """
    def __init__(self, images, targets=None, transforms=None, test=False):
        self.imgs = images
        self.targets = targets
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.imgs[idx]).convert("RGB")) / 255
            if self.transforms: sample = self.transforms(image=img)
            else: sample = {"image": img}

            if not self.test: sample["target"] = self.targets[idx]
            sample["image"] = torch.from_numpy(sample["image"].transpose((2, 0, 1))).float()
            return sample
        except Exception as e:
            print(f"{e} at image {self.imgs[idx]}")
            import pdb; pdb.set_trace()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_targets(test_loader):
    """
    Returns targets from test_loader.
    """
    targets = []
    for i, sample in enumerate(test_loader):
        targets.extend(sample["target"].numpy())
    return targets


def predict(model_name, test_loader, verbose=True):
    """
    Loads best model for "model_name", predicts and returns predictions.

    :param model_name, str: Name of the experiment.
    :param test_loader, torch.utils.data.Dataloader: Dataloader that yields test batches of images and targets.
    :param verbose, bool: If True, print model information when loading weights.
    :return: predictions as numpy array
    """
    arch = model_name.split("_")[0]
    if arch=="resnet50": model = models.resnet50(pretrained=False)
    elif arch=="resnet34": model = models.resnet34(pretrained=False)
    elif arch=="resnet18": model = models.resnet18(pretrained=False)
    else: raise NotImplementedError()

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(classes), bias=True)
    model = model.cuda()
    model.eval()
    model_path = f"trained_models/{model_name}/model_best.pth.tar"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    if verbose: print("{} with loss {:.4f} at epoch {}".format(model_name, checkpoint["best_loss"], checkpoint["epoch"]))
    
    preds = []
    with torch.no_grad():
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            images = sample["image"]
            images = images.cuda(non_blocking=True)

            output = model(images)[:, 0]
            output = torch.sigmoid(output)
            preds.append(output.cpu().numpy())
    # Concatenate predictions with potentially unequal batch size for the last batch
    preds_last = preds[-1]
    preds = np.array(preds[:-1])
    preds = preds.reshape((-1, ))
    preds = np.concatenate((preds, preds_last), axis=0) 
    return preds
