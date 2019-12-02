from utils import *

classes = ["Net_present"]

def main(arch, model_name, train_loader, val_loader, epochs=1000, lr=0.001, patience=2, print_every_x_batches=1000):
    if arch=="resnet50": model = models.resnet50(pretrained=True)
    elif arch=="resnet34": model = models.resnet34(pretrained=True)
    elif arch=="resnet18": model = models.resnet18(pretrained=True)
    else: raise NotImplementedError()

    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(classes), bias=True)
#     model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.cuda()

    criterion = nn.BCEWithLogitsLoss().cuda()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=patience, verbose=True, threshold=0.0001,
                threshold_mode='rel', cooldown=0, eps=1e-08)

    os.makedirs("trained_models/{}".format(model_name), exist_ok=True)
    
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
    print("Best loss of {:.4f} at epoch {}".format(best_loss, best_loss_epoch))
    print("Best Acc@1: {:.4f} at epoch {}".format(best_acc1, best_acc1_epoch))


def train(train_loader, model, criterion, optimizer, epoch, print_every_x_batches):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.4f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
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
        acc1 = ((output>0.5)==target.bool()).sum().float()/output.size(0)

        # measure accuracy and record loss
#         acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
#         top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i>0 and i % print_every_x_batches == 0:
            progress.display(i)
    progress.display(i)

def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    top1 = AverageMeter('Acc@1', ':6.4f')
#     top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            
            images, target = sample["image"], sample["target"]
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)[:, 0]
            loss = criterion(output, target.float())
            output = torch.sigmoid(output)
            acc1 = ((output>0.5)==target.bool()).sum().float()/output.size(0)

            # measure accuracy and record loss
#             acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1, images.size(0))
#             top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        print('Val: Loss {losses.avg:.4f} Acc@1 {top1.avg:.4f}'.format(losses=losses, top1=top1))

    return losses.avg, top1.avg

def save_checkpoint(state, is_best, model_name="test", filename='checkpoint.pth.tar'):
    if is_best: torch.save(state, "trained_models/{}/model_best.pth.tar".format(model_name))


class SalmonDataset(Dataset):
    def __init__(self, images, targets=None, transforms=None, transforms_after_crop=None, infrared=False, 
                 test=False, crop=False, crop_size=None):
        self.imgs = images
        self.targets = targets
        self.transforms = transforms
        self.transforms_after_crop = transforms_after_crop
        self.infrared = infrared
        self.test = test
        self.crop = crop
        self.crop_size = crop_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        try:
            img = np.array(Image.open(self.imgs[idx]).convert("RGB")) / 255
            if self.transforms: sample = self.transforms(image=img)
            else: sample = {"image": img}
                
#             if self.crop=="top_left": sample["image"] = sample["image"][:self.crop_size[0],:self.crop_size[1]]
#             elif self.crop=="top_right": sample["image"] = sample["image"][:self.crop_size[0],-self.crop_size[1]:]
#             elif self.crop=="bottom_right": sample["image"] = sample["image"][-self.crop_size[0]:,-self.crop_size[1]:]
#             elif self.crop=="bottom_left": sample["image"] = sample["image"][-self.crop_size[0]:,:self.crop_size[1]]
                
            if self.transforms_after_crop: sample = self.transforms(image=sample["image"])
            if not self.test: sample["target"] = self.targets[idx]
            sample["image"] = torch.from_numpy(sample["image"].transpose((2, 0, 1))).float()
            return sample
        except Exception as e:
            logging.error("{} at image {}".format(e, self.imgs[idx]), exc_info=True)
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
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_targets(test_loader):
    targets = []
    for i, sample in enumerate(test_loader):
        targets.extend(sample["target"].numpy())
    return targets


def predict(model_name, test_loader, infrared=False, verbose=True):
    arch = model_name.split("_")[0]
    if arch=="resnet50": model = models.resnet50(pretrained=False)
    elif arch=="resnet34": model = models.resnet34(pretrained=False)
    elif arch=="resnet18": model = models.resnet18(pretrained=False)
    else: raise NotImplementedError()
    model.fc = nn.Linear(in_features=model.fc.in_features, out_features=len(classes), bias=True)
#     if infrared: model.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model = model.cuda()
    model.eval()
    model_path = "trained_models/{}/model_best.pth.tar".format(model_name)
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
#             output = torch.softmax(output, dim=1)
            preds.append(output.cpu().numpy())
    preds_last = preds[-1]
    preds = np.array(preds[:-1])
    preds = preds.reshape((-1, ))
    preds = np.concatenate((preds, preds_last), axis=0) 
    return preds
