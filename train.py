import argparse
import logging
import os
import random
import shutil
import datetime
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset, CarvanaDataset, OCTDataset
from evaluate import evaluate
from models import UNet, UNetBackbone, UNet3Plus, DeepLabv3Plus, UTNet, UTNet_Encoderonly
from losses import CELoss, DiceLoss, FocalLoss, LovaszLoss, BoundaryLoss


def seed_all(seed=1000):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_net(
        args,
        net,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False):
    dir_img = args.dir_img
    dir_mask = args.dir_mask
    dir_checkpoint = args.dir_checkpoint + '_' + args.model + '/'
    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Yy%mm%dd_%Hh%Mm%Ss_')
    dir_checkpoint = os.path.join('best_checkpoints', dir_checkpoint, time_str)

    # 1. Create dataset
    try:
        if 'oct' in args.dataset.lower():
            dataset = OCTDataset(dir_img, dir_mask, new_size=args.output_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    if 'oct' in args.dataset.lower():
        train_origin_set = OCTDataset(dir_img, dir_mask, new_size=args.output_size, type='train',
                                      data_aug=args.data_aug)
        val_origin_set = OCTDataset(dir_img, dir_mask, new_size=args.output_size, type='val', data_aug=False)

        train_set, _ = random_split(train_origin_set, [n_train, n_val], generator=torch.Generator().manual_seed(0))
        _, val_set = random_split(val_origin_set, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    else:
        train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    if args.wandb:
        experiment = wandb.init(project=args.model, resume='allow', anonymous='must')
        experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                      val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                      amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5, amsgrad=True)
    elif args.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    else:
        raise NotImplementedError

    if args.lr_scheduler.lower() == 'lambdalr':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif args.lr_scheduler.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.2, patience=30, verbose=True,
                                                         threshold=1e-3,
                                                         threshold_mode="abs")  # goal: maximize Dice score
    elif args.lr_scheduler.lower() == 'cosineannealingwarmrestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=0, verbose=True)
    else:
        raise NotImplementedError

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    global_step = 0
    best_val_score = 0

    ce_loss = CELoss()
    dice_loss = DiceLoss(ignore_index=3)  # 忽略背景类
    lovasz_loss = LovaszLoss(reduction='none')
    boundary_loss = BoundaryLoss(idc=[0, 1, 2])  # 忽略背景类  https://github.com/LIVIAETS/surface-loss/issues/3
    alpha = 0.6

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']
                dist_map_label = batch['distmap']

                assert images.shape[1] == net.in_channels, \
                    f'Network has been defined with {net.in_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                dist_map_label = dist_map_label.to(device=device)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = ce_loss(masks_pred, true_masks)
                    # print("celoss", celoss)

                    if args.diceloss:
                        diceloss = dice_loss(masks_pred, true_masks)
                        loss += diceloss
                    if args.lovaszloss:
                        lovaszloss = lovasz_loss(masks_pred, true_masks)
                        loss += lovaszloss
                        # print("lovaszloss", lovaszloss)
                    if args.boundaryloss:
                        pred_probs = F.softmax(masks_pred, dim=1)
                        boundaryloss = boundary_loss(pred_probs, dist_map_label)
                        loss = alpha * loss + (1 - alpha) * boundaryloss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                if args.wandb:
                    experiment.log({
                        'train loss': loss.item(),
                        'step': global_step,
                        'epoch': epoch
                    })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (10 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        if args.wandb:
                            histograms = {}
                            for tag, value in net.named_parameters():
                                tag = tag.replace('/', '.')
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(args, net, val_loader, device)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        if args.wandb:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(
                                        torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })

                        # save best model
                        if best_val_score < val_score:
                            best_val_score = val_score
                            if save_checkpoint:
                                if os.path.exists(dir_checkpoint):
                                    shutil.rmtree(dir_checkpoint)
                                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                torch.save(net.state_dict(),
                                           dir_checkpoint + '/best_checkpoint_dice{:.5f}.pth'.format(best_val_score))
                                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', metavar='M', type=str, default='unet', help='Name of model')
    parser.add_argument('--dataset', '-d', metavar='D', type=str, default='OCT', help='Name of dataset')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=70, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--val_ratio', '-v', metavar='V', type=float, default=0.2,
                        help='Percent of the data that is used as validation')

    parser.add_argument('--diceloss', action='store_true', default=False, help='Use Dice Loss')
    parser.add_argument('--lovaszloss', action='store_true', default=False, help='Use Lovasz Loss')
    parser.add_argument('--boundaryloss', action='store_true', default=False, help='Use Boundary Loss')
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--lr_scheduler', default='LambdaLR',
                        type=str, help='name of lr scheduler used in training')

    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')

    parser.add_argument('--output_size', type=int, default=128, help='Output size of the images')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    parser.add_argument('--dir_img', type=str, default='../Train/Image/', help='images directory')
    parser.add_argument('--dir_mask', type=str, default='../Train/Layer_Masks/', help='masks directory')
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints', help='model saved directory')

    parser.add_argument('--data_aug', action='store_true', default=False, help='Use extra data augmentation')
    parser.add_argument('--wandb', action='store_true', default=False, help='Show results in training process')

    parser.add_argument('--device', default='0', type=str)

    return parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=1000)
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Using device {device}')

    if args.model.lower() == 'unet':
        net = UNet(in_channels=3, num_classes=args.classes, bilinear=args.bilinear)
    elif args.model.lower() == 'unet_backone':
        net = UNetBackbone(num_classes=args.classes, pretrained=False, backbone='vgg')
    elif args.model.lower() == 'unet3plus':
        net = UNet3Plus(in_channels=3, num_classes=args.classes, is_batchnorm=True, is_deepsup=False, is_CGM=False)
    elif args.model.lower() == 'deeplabv3plus':
        net = DeepLabv3Plus(num_classes=args.classes, backbone="resnet101", pretrained=True, downsample_factor=16)
    else:
        raise NotImplementedError

    logging.info(f'Network: {args.model}\n'
                 f'\t{net.in_channels} input channels\n'
                 f'\t{net.num_classes} output channels (classes)\n')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(
            args=args,
            net=net,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val_ratio,
            amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
