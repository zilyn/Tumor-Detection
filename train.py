import os
import yaml
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from glob import glob
from tqdm import tqdm
from collections import OrderedDict
from source.utils import iou_score, AverageMeter
from albumentations.augmentations import transforms
from albumentations.augmentations import geometric
from sklearn.model_selection import train_test_split
from albumentations.core.composition import Compose, OneOf

from source.network import UNetPP
from source.dataset import DataSet


def train(deep_sup, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for input, target, _ in train_loader:
        input = input.to(device)
        target = target.to(device)

        # compute output
        if deep_sup:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(deep_sup, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.to(device)
            target = target.to(device)

            # compute output
            if deep_sup:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


if __name__ == "__main__":

    with open("config.yaml") as f:
        config = yaml.load(f)

    extn = config["extn"]
    epochs = config["epochs"]
    log_path = config["log_path"]
    mask_path = config["mask_path"]
    image_path = config["image_path"]
    model_path = config["model_path"]

    log = OrderedDict([
        ('epoch', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0

    extn_ = f"*{extn}"
    img_ids = glob(os.path.join(image_path, extn_))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2)

    train_transform = Compose([
    geometric.rotate.RandomRotate90(),
    geometric.transforms.Flip(),
    OneOf([
        transforms.HueSaturationValue(),
        transforms.RandomBrightnessContrast()
    ], p=1),
    geometric.resize.Resize(256, 256),
    transforms.Normalize(),
    ])

    val_transform = Compose([
    geometric.resize.Resize(256, 256),
    transforms.Normalize(),
    ])

    train_dataset = DataSet(
        img_ids=train_img_ids,
        img_dir=image_path,
        mask_dir=mask_path,
        img_ext=extn,
        mask_ext=extn,
        transform=train_transform)

    val_dataset = DataSet(
        img_ids=val_img_ids,
        img_dir=image_path,
        mask_dir=mask_path,
        img_ext=extn,
        mask_ext=extn,
        transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False)

    model = UNetPP(1, 3, True)
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.BCEWithLogitsLoss()
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-4)

    for epoch in range(epochs):
        print(f'Epoch [{epoch}/{epochs}]')

        # train for one epoch
        train_log = train(True, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(True, val_loader, model, criterion)

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                  % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv(log_path, index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), model_path)
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
