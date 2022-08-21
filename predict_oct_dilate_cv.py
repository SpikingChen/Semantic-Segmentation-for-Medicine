'''
Description : 使用膨胀预测的输出实验结果
'''
import sys
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import cv2
from utils.nvidia_info import get_free_device_ids
import datetime
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from statistics import mode

from utils.data_loading import OCTDataset, InferenceDataset
from models import UNet, UNetBackbone, UNet3Plus, DeepLabv3Plus

torch.backends.cudnn.benchmark = True


def create_zeros_png(image_w, image_h):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以被滑动窗口整除；
        2. 膨胀预测：预测时，对每个(1024,1024)窗口，每次只保留中心(512,512)区域预测结果，每次滑窗步长为512，使预测结果不交叠；
    '''
    new_h, new_w = 896, 1152  # 填充右边界
    zeros = (new_h, new_w)  # 填充空白边界
    zeros = np.ones(zeros, np.uint8) * 255
    return zeros


def tta_forward(dataloader, all_models, png_shape, device=None, policy='average'):
    image_w, image_h = png_shape
    predict_png = create_zeros_png(image_w, image_h)

    with torch.no_grad():
        for (image, pos_list) in tqdm(dataloader):
            predict_lists = []
            for model in all_models:
                # forward --> predict
                image = image.cuda(device)

                predict_1 = model(image)

                predict_2 = model(torch.flip(image, [-1]))
                predict_2 = torch.flip(predict_2, [-1])

                predict_3 = model(torch.flip(image, [-2]))
                predict_3 = torch.flip(predict_3, [-2])

                predict_4 = model(torch.flip(image, [-1, -2]))
                predict_4 = torch.flip(predict_4, [-1, -2])

                predict_list = predict_1 + predict_2 + predict_3 + predict_4

                if policy == 'average':  # 取每个模型预测的平均结果作为最终结果
                    predict_lists.append(predict_list)
                elif policy == 'voting':   # 模型预测结果作为票数，得票多的为最终结果
                    predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w
                    predict_lists.append(predict_list)
                else:
                    raise NotImplementedError

            batch_size = predict_list.shape[0]  # batch大小
            output_h = predict_list.shape[-2]
            output_w = predict_list.shape[-1]

            if policy == 'average':
                predict_list = torch.zeros(batch_size, 4, output_h, output_w).to(predict_list.device)
                for single_model_predict_result in predict_lists:
                    predict_list += single_model_predict_result

                predict_list = torch.argmax(predict_list.cpu(), 1).byte().numpy()  # n x h x w

            elif policy == 'voting':
                predict_results = []
                for single_model_predict_result in predict_lists:
                    single_model_predict_result = np.reshape(single_model_predict_result, (batch_size, -1))
                    predict_results.append(single_model_predict_result)

                predict_list = np.stack(predict_results, axis=1)
                predict_results = []
                for i in range(batch_size):
                    one_image_predict = np.array([mode(v) for v in predict_list[i, :, :].T])
                    one_image_predict = np.reshape(one_image_predict, (output_h, output_w))

                    predict_results.append(one_image_predict)

                predict_list = np.stack(predict_results, axis=0)

            else:
                raise NotImplementedError

            for i in range(batch_size):

                predict = predict_list[i]
                predict[predict == 1] = 80
                predict[predict == 2] = 160
                predict[predict == 3] = 255

                pos = pos_list[i, :]
                [topleft_x, topleft_y, buttomright_x, buttomright_y] = pos

                if (buttomright_x - topleft_x) == 512 and (buttomright_y - topleft_y) == 512:
                    # 每次预测只保留图像中心(464,486)区域预测结果
                    predict_png[topleft_y + 48:buttomright_y - 48, topleft_x + 26:buttomright_x - 26] = predict[
                                                                                                        48:512 - 48,
                                                                                                        26:512 - 26
                                                                                                        ]
                else:
                    raise ValueError(
                        "target_size!=512， Got {},{}".format(buttomright_x - topleft_x, buttomright_y - topleft_y))

    h, w = predict_png.shape
    predict_png = predict_png[48:h - 48, 26:w - 26]  # 去除整体外边界

    return predict_png


def label_vis(label, img=None, alpha=0.5):
    '''
    :param label:原始标签
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    r = np.where(label == 0, 255, 0)
    g = np.where(label == 80, 255, 0)
    b = np.where(label == 160, 255, 0)

    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0]
    anno_vis[:, :, 1] = anno_vis[:, :, 1]
    anno_vis[:, :, 2] = anno_vis[:, :, 2]
    if img is None:
        return anno_vis
    else:
        overlapping = cv2.addWeighted(img, alpha, anno_vis, 1 - alpha, 0)
        return overlapping


def get_args():
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--model', '-m', metavar='M', type=str, default='UNet', help='Name of model')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_cv/')
    parser.add_argument('--root_path', type=str, default='./preprocess_val_data')
    parser.add_argument('--save_dir', type=str, default='./predict_results')
    parser.add_argument('--batch_size', '-b', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--policy', type=str, default='average')

    return parser.parse_args()


def select_model(args):
    if args.model.lower() == 'unet':
        model = UNet(in_channels=3, num_classes=4, bilinear=False)
    elif args.model.lower() == 'unet_backone':
        model = UNetBackbone(num_classes=4, pretrained=False, backbone='vgg')
    elif args.model.lower() == 'unet3plus':
        model = UNet3Plus(in_channels=3, num_classes=4, is_batchnorm=True, is_deepsup=False, is_CGM=False)
    elif args.model.lower() == 'deeplabv3plus':
        model = DeepLabv3Plus(num_classes=4, backbone="mobilenet", pretrained=True, downsample_factor=16)
    else:
        raise NotImplementedError

    return model


if __name__ == "__main__":

    args = get_args()

    # get free_device
    free_device_ids = get_free_device_ids()
    max_num_devices = 8
    if len(free_device_ids) >= max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    if len(free_device_ids) == 0:
        free_device_ids = [0]

    master_device = free_device_ids[0]

    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Yy%mm%dd_%Hh%Mm%Ss_')
    model_tag = args.model.lower()
    save_dir = os.path.join('prediction_results', args.save_dir, time_str + model_tag, 'Layer_Segmentations')
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = args.checkpoint_path
    all_models = list()
    for checkpoint_dir_name in os.listdir(checkpoint_path):
        checkpoint_dir = os.path.join(checkpoint_path, checkpoint_dir_name)
        for checkpoint_name in os.listdir(checkpoint_dir):
            checkpoint = os.path.join(checkpoint_dir, checkpoint_name)

            model = select_model(args)
            model.load_state_dict(torch.load(checkpoint, map_location='cpu'))

            model = nn.DataParallel(model, device_ids=free_device_ids).cuda(master_device)

            model = model.eval()

            all_models.append(model)

    root_path = args.root_path
    image_dir = os.path.join(root_path, 'image')  # 测试图像路径
    for filename in os.listdir(root_path):
        if filename[-4:] == '.csv':
            csv_file = os.path.join(root_path, filename)

            test_set = InferenceDataset(image_dir, csv_file)
            loader_args = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
            test_loader = DataLoader(test_set, shuffle=True, **loader_args)
            predict_png = tta_forward(test_loader, all_models, png_shape=(800, 1100), device=master_device,
                                      policy=args.policy)

            pil_image = Image.fromarray(predict_png)
            pil_image.save(os.path.join(save_dir, filename[:-4] + ".png"))
            print("{} saved".format(filename[:-4] + ".png"))
