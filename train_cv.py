import argparse
import logging
import os
from pathlib import Path
import random
import numpy as np
import shutil
import datetime

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

import wandb

from utils.nvidia_info import get_free_device_ids
from utils.data_loading import BasicDataset, OCTDataset
from evaluate import evaluate
from models import UNet, UNetBackbone, UNet3Plus, DeepLabv3Plus
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
        train_loader,
        val_loader,
        save_path,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        amp: bool = False,
        ki: int = 0
):

    master_device = "cuda:" + str(device)
    device = torch.device(master_device)

    n_val = len(val_loader)
    n_train = len(train_loader)

    # (Initialize logging)
    experiment = wandb.init(project=args.model + '_kfold' + str(ki), resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device}
        Mixed Precision: {amp}
    ''')

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    if args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif args.optim.lower() == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    else:
        raise NotImplementedError

    if args.lr_scheduler.lower() == 'lambdalr':
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
    elif args.lr_scheduler.lower() == 'reducelronplateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    else:
        raise NotImplementedError

    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

    criterion = nn.CrossEntropyLoss()

    ce_loss = CELoss().to(device)
    dice_loss = DiceLoss().to(device)

    global_step = 0
    best_val_score = 0

    # Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0

        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']

            assert images.shape[1] == net.in_channels, \
                f'Network has been defined with {net.in_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                celoss = ce_loss(masks_pred, true_masks)
                diceloss = dice_loss(masks_pred, true_masks, ignore_index=3)

                # loss = celoss + diceloss
                loss = celoss

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            experiment.log({
                'train loss': loss.item(),
                'step': global_step,
                'epoch': epoch
            })

            # Evaluation round
            division_step = (n_train // (batch_size))
            if division_step > 0:
                if global_step % division_step == 0:
                    logging.info(
                        "KFold [%d] Training Epoch: [%d] / [%d], step: %d, train loss: %0.6f"
                        % (ki, epoch, epochs, global_step, loss.item())
                    )
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                    val_score = evaluate(args, net, val_loader, device)
                    scheduler.step(val_score)

                    logging.info('Validation Dice score: {}'.format(val_score))
                    experiment.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'validation Dice': val_score,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].float().cpu()),
                            'pred': wandb.Image(torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu())
                        },
                        'step': global_step,
                        'epoch': epoch,
                        **histograms
                    })

                    # save best model
                    if best_val_score < val_score:
                        best_val_score = val_score
                        if save_checkpoint:
                            if os.path.exists(save_path):
                                shutil.rmtree(save_path)
                            Path(save_path).mkdir(parents=True, exist_ok=True)
                            torch.save(net.state_dict(),
                                       save_path + '/best_checkpoint_dice{:.5f}.pth'.format(best_val_score))
                            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--model', '-m', metavar='M', type=str, default='UNet', help='Name of model')
    parser.add_argument('--dataset', '-d', metavar='D', type=str, default='OCT', help='Name of dataset')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--val_ratio', '-v', metavar='V', type=float, default=0.2,
                        help='Percent of the data that is used as validation')

    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--lr_scheduler', default='LambdaLR',
                        type=str, help='name of lr scheduler used in training')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.1, help='Downscaling factor of the images')
    parser.add_argument('--output_size', type=int, default=128, help='Output size of the images')

    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')

    parser.add_argument("--parallel", '-p', type=str, default='DP', help='choose DP or DDP')
    parser.add_argument('--device', type=int, nargs='+', default=None, help="list of device_id, e.g. [0,1,2]")

    parser.add_argument('--dir_img', type=str, default='../Train/Image/', help='images directory')
    parser.add_argument('--dir_mask', type=str, default='../Train/Layer_Masks/', help='masks directory')
    parser.add_argument('--dir_checkpoint', type=str, default='./checkpoints', help='model saved directory')

    return parser.parse_args()


def main(args):
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    if args.device:  # 传入命令指定 device id
        free_device_ids = args.device
    else:
        free_device_ids = get_free_device_ids()

    # get free_device
    max_num_devices = 8
    if len(free_device_ids) >= max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    if len(free_device_ids) == 0:
        free_device_ids = [0]

    if 3 in free_device_ids:
        free_device_ids.remove(3)

    master_device = free_device_ids[0]

    logging.info(f'Using device {free_device_ids}')

    dir_img = args.dir_img
    dir_mask = args.dir_mask
    dir_checkpoint = args.dir_checkpoint + '_' + args.model.lower() + '/'
    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Yy%mm%dd_%Hh%Mm%Ss_')
    dir_checkpoint = os.path.join('best_checkpoints', dir_checkpoint, time_str)

    # Create dataset
    try:
        dataset = OCTDataset(dir_img, dir_mask, new_size=args.output_size)
    except (AssertionError, RuntimeError):
        dataset = BasicDataset(dir_img, dir_mask)

    loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # 5 fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for ki, (train_index, val_index) in enumerate(kf.split(dataset)):
        save_path = os.path.join(dir_checkpoint, 'kfold_{}'.format(ki))

        # Change here to adapt to your data
        # n_channels=3 for RGB images
        # n_classes is the number of probabilities you want to get per pixel
        if args.model.lower() == 'unet':
            net = UNet(in_channels=args.in_channels, num_classes=args.classes, bilinear=args.bilinear)
        elif args.model.lower() == 'unet_backone':
            net = UNetBackbone(num_classes=args.classes, pretrained=False, backbone='vgg')
        elif args.model.lower() == 'unet3plus':
            net = UNet3Plus(in_channels=args.in_channels, num_classes=args.classes, is_batchnorm=True, is_deepsup=False,
                            is_CGM=False)
        elif args.model.lower() == 'deeplabv3plus':
            net = DeepLabv3Plus(num_classes=args.classes, backbone="mobilenet", pretrained=True, downsample_factor=16)
        else:
            raise NotImplementedError

        logging.info(f'Network: {args.model.lower()}\n'
                     f'\t{net.in_channels} input channels\n'
                     f'\t{net.num_classes} output channels (classes)\n')

        if args.load:
            net.load_state_dict(torch.load(args.load, map_location='cpu'))
            logging.info(f'Model loaded from {args.load}')

        net.cuda(master_device)

        train_subset = torch.utils.data.dataset.Subset(dataset, train_index)
        val_subset = torch.utils.data.dataset.Subset(dataset, val_index)

        train_loader = DataLoader(train_subset, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_subset, shuffle=False, drop_last=False, **loader_args)

        try:
            train_net(
                args=args,
                net=net,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                train_loader=train_loader,
                val_loader=val_loader,
                device=master_device,
                val_percent=args.val_ratio,
                amp=args.amp,
                save_path=save_path,
                ki=ki)
        except KeyboardInterrupt:
            torch.save(net.state_dict(), 'INTERRUPTED.pth')
            logging.info('Saved interrupt')
            raise


if __name__ == '__main__':
    seed_all(seed=1000)
    args = get_args()

    main(args)
