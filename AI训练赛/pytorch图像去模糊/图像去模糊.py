import torch

import numpy as np

import os
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tensorboardX import SummaryWriter
from torchsummary import summary
from torch.optim import lr_scheduler
from torch.utils import data
from torchvision import transforms
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Config():
    def __init__(self, name="Configs"):
        # train set
        self.data_dir = 'datasets/train'  # 训练集目录
        self.patch_size = 256  # 输入模型的patch的尺寸
        self.batch_size = 2  # 16 # 训练时每个batch中的样本个数
        self.n_threads = 1  # 用于加载数据的线程数

        # test set
        self.test_data_dir = 'datasets/test'  # 测试集目录
        self.test_batch_size = 1  # 测试时的 batch_size

        # model
        self.multi = True  # 模型采用多尺度方法True
        self.skip = True  # 模型采用滑动连接方法
        self.n_resblocks = 3  # 9  # resblock的个数
        self.n_feats = 8  # 64  #feature map的个数

        # optimization
        self.lr = 1e-4  # 初始学习率
        self.epochs = 1  # 800 # 训练epoch的数目
        self.lr_step_size = 600  # 采用步进学习率策略所用的 step_size
        self.lr_gama = 0.1  # 每 lr_step_size后，学习率变成 lr * lr_gama

        # global
        self.name = name  # 配置的名称
        self.save_dir = 'temp/result'  # 保存训练过程中所产生数据的目录
        self.save_cp_dir = 'temp/models'  # 保存 checkpoint的目录
        self.imgs_dir = 'datasets/pictures'  # 此 notebook所需的图片目录

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.save_cp_dir):
            os.makedirs(self.save_cp_dir)


#         if not os.path.exists(self.data_dir):
#             os.makedirs(self.data_dir)
#         if not os.path.exists(self.test_data_dir):
#             os.makedirs(self.test_data_dir)

args = Config(name="image-deblurring")


def argment(img_input, img_target):
    degree = random.choice([0, 90, 180, 270])

    img_input = transforms.functional.rotate(img_input, 1)
    img_target = transforms.functional.adjust_gamma(img_target, 1)

    sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
    img_input = transforms.functional.adjust_saturation(img_input, sat_factor)

    return img_input, img_target


def getPatch(img_input, img_target, patch_size):
    w, h = img_input.size
    p = patch_size
    x = random.randrange(0, w - p + 1)
    y = random.randrange(0, h - p + 1)

    img_input = img_input.crop((x, y, x + p, y + p))
    img_target = img_target.crop((x, y, x + p, y + p))

    return img_input, img_target


class Gopro(data.Dataset):
    def __init__(self, data_dir, patch_size=256, is_train=False, multi=True):
        super(Gopro, self).__init__()

        self.is_train = is_train  # 是否是训练集
        self.patch_size = patch_size  # 训练时 patch的尺寸
        self.multi = multi  # 是否采用多尺度因子，默认采用

        self.sharp_file_paths = []
        sub_folders = os.listdir(data_dir)
        print('sub_folders')
        print(sub_folders)

        for folder_name in sub_folders:
            sharp_sub_folder = os.path.join(data_dir, folder_name, 'sharp')
            sharp_file_names = os.listdir(sharp_sub_folder)

            for file_name in sharp_file_names:
                sharp_file_path = os.path.join(sharp_sub_folder, file_name)
                self.sharp_file_paths.append(sharp_file_path)

        self.n_samples = len(self.sharp_file_paths)

    def get_img_pair(self, idx):
        sharp_file_path = self.sharp_file_paths[idx]
        blur_file_path = sharp_file_path.replace("sharp", "blur")

        img_input = Image.open(blur_file_path).convert('RGB')
        img_target = Image.open(sharp_file_path).convert('RGB')

        return img_input, img_target

    def __getitem__(self, idx):
        img_input, img_target = self.get_img_pair(idx)

        if self.is_train:
            img_input, img_target = getPatch(img_input, img_target, self.patch_size)
            img_input, img_target = argment(img_input, img_target)

        # 转换为 tensor类型
        input_b1 = transforms.ToTensor()(img_input)
        target_s1 = transforms.ToTensor()(img_target)

        H = input_b1.size()[1]
        W = input_b1.size()[2]

        if self.multi:
            input_b1 = transforms.ToPILImage()(input_b1)
            target_s1 = transforms.ToPILImage()(target_s1)

            input_b2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(input_b1))
            input_b3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(input_b1))

            # 只对训练集进行数据增强
            if self.is_train:
                target_s2 = transforms.ToTensor()(transforms.Resize([int(H / 2), int(W / 2)])(target_s1))
                target_s3 = transforms.ToTensor()(transforms.Resize([int(H / 4), int(W / 4)])(target_s1))
            else:
                target_s2 = []
                target_s3 = []

            input_b1 = transforms.ToTensor()(input_b1)
            target_s1 = transforms.ToTensor()(target_s1)

            return {
                'input_b1': input_b1,  # 参照下文的网络结构，输入图像的尺度 1
                'input_b2': input_b2,  # 输入图像的尺度 2
                'input_b3': input_b3,  # 输入图像的尺度 3
                'target_s1': target_s1,  # 目标图像的尺度 1
                'target_s2': target_s2,  # 目标图像的尺度 2
                'target_s3': target_s3  # 目标图像的尺度 3
            }
        else:
            return {'input_b1': input_b1, 'target_s1': target_s1}

    def __len__(self):
        return self.n_samples


def get_dataset(data_dir, patch_size=None,
                batch_size=1, n_threads=1,
                is_train=False, multi=False):
    # Dataset实例化
    dataset = Gopro(data_dir, patch_size=patch_size,
                    is_train=is_train, multi=multi)

    # 利用封装好的 dataloader 接口定义训练过程的迭代器
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             drop_last=True, shuffle=is_train,
                                             num_workers=int(n_threads))
    return dataloader


data_loader = get_dataset(args.data_dir,
                          patch_size=args.patch_size,
                          batch_size=args.batch_size,
                          n_threads=args.n_threads,
                          is_train=True,
                          multi=args.multi
                          )


def default_conv(in_channels, out_channels, kernel_size, bias):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias)


class UpConv(nn.Module):
    def __init__(self):
        super(UpConv, self).__init__()
        self.body = nn.Sequential(default_conv(3, 12, 3, True),
                                  nn.PixelShuffle(2),
                                  nn.ReLU(inplace=True))

    def forward(self, x):
        return self.body(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResidualBlock, self).__init__()

        modules_body = [
            default_conv(n_feats, n_feats, 3, bias=True),
            nn.ReLU(inplace=True),
            default_conv(n_feats, n_feats, 3, bias=True)
        ]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SingleScaleNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, is_skip, n_channels=3):
        super(SingleScaleNet, self).__init__()
        self.is_skip = is_skip

        modules_head = [
            default_conv(n_channels, n_feats, 5, bias=True),
            nn.ReLU(inplace=True)
        ]

        modules_body = [ResidualBlock(n_feats) for _ in range(n_resblocks)]
        modules_tail = [default_conv(n_feats, 3, 5, bias=True)]

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        if self.is_skip:
            res += x

        res = self.tail(res)
        return res


class MultiScaleNet(nn.Module):
    def __init__(self, n_feats, n_resblocks, is_skip):
        super(MultiScaleNet, self).__init__()

        self.scale3_net = SingleScaleNet(n_feats,
                                         n_resblocks,
                                         is_skip,
                                         n_channels=3)
        self.upconv3 = UpConv()
        self.scale2_net = SingleScaleNet(n_feats,
                                         n_resblocks,
                                         is_skip,
                                         n_channels=6)
        self.scale2_net = UpConv()

        self.scale1_net = SingleScaleNet(n_feats,
                                         n_resblocks,
                                         is_skip,
                                         n_channels=6)

    def forward(self, mulscale_input):
        input_b1, input_b2, input_b3 = mulscale_input

        output_l3 = self.scale3_net(input_b3)
        output_l3_up = self.upconv3(output_l3)

        output_l2 = self.scale2_net(torch.cat((input_b2, output_l3_up), 1))
        output_l2_up = self.upconv2(output_l2)

        output_l1 = self.scale2_net(torch.cat((input_b1, output_l2_up), 1))

        return output_l1, output_l2, output_l3


if args.multi:
    my_model = MultiScaleNet(n_feats=args.n_feats,
                             n_resblocks=args.n_resblocks,
                             is_skip=args.skip)
else:
    my_model = SingleScaleNet(n_feats=args.n_feats,
                              n_resblocks=args.n_resblocks,
                              is_skip=args.skip)

if torch.cuda.is_available():
    my_model.cuda()
    loss_function = nn.MSELoss().cuda()
else:
    loss_function = nn.MSELoss()

optimizer = optim.Adam(my_model.parameters(), lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, args.lr_step_size, args.lr_gama)

writer = SummaryWriter(os.path.join(args.save_dir, "temp/logs/"))

bar_format = '{desc}{percentage:3.0f}% | [{elapsed}<{remaining},{rate_fmt}]'  # 进度条格式
if __name__ == '__main__':
    for epoch in range(args.epochs):
        total_loss = 0
        batch_bar = tqdm(data_loader, bar_format=bar_format)  # 利用tqdm动态显示训练过程
        print('batch_bar')
        print(batch_bar)

        for batch, images in enumerate(batch_bar):
            print('batch')
            print(batch)
            my_model.train()
            curr_batch = epoch * data_loader.__len__() + batch  # 当前batch在整个训练过程中的索引

            input_b1 = images['input_b1'].to(device)  # 原始输入图像
            target_s1 = images['target_s1'].to(device)  # 目标非模糊图片

            if args.multi:
                input_b2 = images['input_b2'].to(device)  # level-2 尺度
                target_s2 = images['target_s2'].to(device)

                input_b3 = images['input_b3'].to(device)  # level-3 尺度
                target_s3 = images['target_s3'].to(device)
                output_l1, output_l2, output_l3 = my_model((input_b1, input_b2, input_b3))

                # 损失函数
                loss = (loss_function(output_l1, target_s1) + loss_function(output_l2, target_s2) + loss_function(
                    output_l3, target_s3)) / 3

            else:
                output_l1 = my_model(input_b1)
                loss = loss_function(output_l1, target_s1)

            my_model.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新权值
            total_loss += loss.item()

            print_str = "|".join([
                "epoch:%3d/%3d" % (epoch + 1, args.epochs),
                "batch:%3d/%3d" % (batch + 1, data_loader.__len__()),
                "loss:%.5f" % (loss.item()),
            ])
            batch_bar.set_description(print_str, refresh=True)  # 更新进度条

            writer.add_scalar('train/batch_loss', loss.item(), curr_batch)

        batch_bar.close()
        scheduler.step()  # 调整学习率
        loss = total_loss / (batch + 1)

        writer.add_scalar('train/batch_loss', loss, epoch)
        torch.save(my_model.state_dict(), os.path.join(args.save_cp_dir, f'Epoch_{epoch}.pt'))  # 保存每个 epoch 的参数
    #     torch.save(my_model.state_dict(),os.path.join(args.save_cp_dir, f'Epoch_lastest.pt')) # 保存最新的参数
