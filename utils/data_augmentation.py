import cv2
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np


def oct_data_aug(imgs, masks, size):
    # 标准化格式
    imgs = np.array(imgs)
    masks = np.array(masks).astype(np.uint8)

    # 注意mask新填充的像素不能为黑色0（对应了目标的label），应置为白色255（背景类）
    masks[masks == 0] = 25
    masks[masks == 255] = 0
    masks[masks == 25] = 255

    # print('imgs shape',imgs.shape)

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)  # 设定随机函数,50%几率扩增,or

    seq = iaa.Sequential(
        [
            iaa.CropToFixedSize(width=size, height=size),

            iaa.Fliplr(0.5),  # 50%图像进行水平翻转
            iaa.Flipud(0.5),  # 50%图像做垂直翻转

            # sometimes(iaa.Affine(  # 对一部分图像做仿射变换
            #     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # 图像缩放为80%到120%之间
            #     translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # 平移±20%之间
            #     rotate=(-45, 45),  # 旋转±45度之间
            #     shear=(-16, 16),  # 剪切变换±16度，（矩形变平行四边形）
            #     order=[0, 1],  # 使用最邻近差值或者双线性差值
            #     cval=(0, 255),
            #     mode=ia.ALL,  # 边缘填充
            # )),

            # 分段仿射
            # iaa.PiecewiseAffine(scale=(0.01, 0.05)),

            # 伽马变换
            # iaa.GammaContrast((0.5, 2.0)),
            # iaa.GammaContrast(1.9),

            # 随机弹性变换
            iaa.ElasticTransformation(alpha=300, sigma=30),

            # 使用下面的0个到5个之间的方法去增强图像
            iaa.SomeOf((0, 5),
                       [
                           # 锐化处理
                           # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.25)),

                           # 用高斯模糊，均值模糊，中值模糊中的一种增强
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),  # 核大小2~7之间，k=((5, 7), (1, 3))时，核高度5~7，宽度1~3
                               iaa.MedianBlur(k=(3, 11)),
                           ]),

                           # 加入高斯噪声
                           iaa.AdditiveGaussianNoise(
                               loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5
                           ),
                       ],
                       random_order=True    # 随机的顺序把这些操作用在图像上
                       )
        ],
        random_order=True  # 随机的顺序把这些操作用在图像上
    )

    seq_det = seq.to_deterministic()  # 确定一个数据增强的序列
    # print('imgs.shape',imgs.shape)
    segmaps = ia.SegmentationMapsOnImage(masks, shape=masks.shape)  # 分割标签格式
    image_aug, segmaps_aug_iaa = seq_det(image=imgs, segmentation_maps=segmaps)  # 将方法同步应用在图像和分割标签上，
    segmap_aug = segmaps_aug_iaa.get_arr().astype(np.uint8)  # 转换成np类型

    segmap_aug[segmap_aug == 255] = 25
    segmap_aug[segmap_aug == 0] = 255
    segmap_aug[segmap_aug == 25] = 0

    return image_aug, segmap_aug


if __name__ == "__main__":
    imgs = cv2.imread("./img.png")
    masks = cv2.imread("./gt.png")

    # images_aug, segmaps_aug = data_aug(imgs, masks, segs=None)
    images_aug, segmaps_aug = oct_data_aug(imgs, masks, 512)

    cv2.imwrite("./image.png", images_aug)
    cv2.imwrite("./mask.png", segmaps_aug)
