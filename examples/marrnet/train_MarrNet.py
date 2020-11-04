from os import makedirs
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks import ViewAsLinear
from uresnet import Net as Uresnet
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.init as init
import os

# how to fix nan problem??


cuda_5 = torch.device('cuda:5')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
    elif classname.find('Linear') != -1 and classname.find("ViewAsLinear") == 1:
        init.xavier_normal_(m.weight.data)


class DealDataset(Dataset):

    def __init__(self):
        depth = np.load("/mnt/zhengwen/model_synthesis/SF_temp/02691156_depth.npz.npy")
        RGB_in = np.load("/mnt/zhengwen/model_synthesis/SF_temp/02691156_rgb.npz.npy")[:, :, :-1, :, :]
        normal = np.load("/mnt/zhengwen/model_synthesis/SF_temp/02691156_norma.npz.npy")
        silhou = np.load("/mnt/zhengwen/model_synthesis/SF_temp/02691156_rgb.npz.npy")[:, :, -1:, :, :]
        self.depth_gt = torch.from_numpy(depth.reshape((-1, 1, 64, 64)))
        self.normal_gt = torch.from_numpy(normal.reshape(-1, 3, 64, 64))
        self.silhou_gt = torch.from_numpy(silhou.reshape(-1, 1, 64, 64))
        self.RGB_in = torch.from_numpy(RGB_in.reshape(-1, 3, 64, 64))
        self.len = self.depth_gt.shape[0]

    def __getitem__(self, index):
        return {
            "depth": self.depth_gt[index],
            "normal": self.normal_gt[index],
            "silhou": self.silhou_gt[index],
            "RGB": self.RGB_in[index]
        }

    def __len__(self):
        return self.len


SCALE_25D = 100
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


def postprocess(tensor, scale_25d):
    scaled = tensor / scale_25d
    return scaled


class Net(Uresnet):
    def __init__(self, *args, pred_depth_minmax=True):
        super().__init__(*args)
        self.pred_depth_minmax = pred_depth_minmax
        if self.pred_depth_minmax:
            module_list = nn.Sequential(
                nn.Conv2d(512, 512, 2, stride=2),
                nn.Conv2d(512, 512, 4, stride=1),
                ViewAsLinear(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)
            )
            self.decoder_minmax = module_list

    def forward(self, input):
        x = input
        out_dict = super().forward(x)
        # if self.pred_depth_minmax:
        #     out_dict['depth_minmax'] = self.decoder_minmax(self.encoder_out)
        return out_dict


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        im_size = 256
        rgb_jitter_d = 0.4
        rgb_light_noise = 0.1
        silhou_thres = 0.999
        pred_silhou_thres = 0.3
        scale_25d = 100
        learning_rate = 0.00001
        self.model = Net(
            [3, 1, 1],
            ['normal', 'depth', 'silhou'],
        )
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )
        self.criterion = nn.functional.mse_loss

    def forward(self, x):
        output = self.model(x)
        return output

    def compute_loss(self, pred, gt):
        pred_normal = pred['normal']
        pred_depth = pred['depth']
        pred_silhou = pred['silhou']
        is_fg = gt['silhou'] != 0  # excludes background
        is_fg_full = is_fg.expand_as(pred_normal)
        loss_normal = self.criterion(
            pred_normal[is_fg_full], gt['normal'][is_fg_full]
        )
        loss_depth = self.criterion(
            pred_depth[is_fg], gt['depth'][is_fg]
        )
        loss_silhou = self.criterion(pred_silhou, gt['silhou'])
        loss = loss_normal + loss_depth + loss_silhou
        loss_data = {}
        loss_data['loss'] = loss.mean().item()
        loss_data['normal'] = loss_normal.mean().item()
        loss_data['depth'] = loss_depth.mean().item()
        loss_data['silhou'] = loss_silhou.mean().item()
        # if self.opt.pred_depth_minmax:
        #     w_minmax = (256 ** 2) / 2  # matching scale of pixel predictions very roughly
        #     loss_depth_minmax = w_minmax * self.criterion(
        #         pred['depth_minmax'],
        #         self._gt.depth_minmax
        #     )
        #     loss += loss_depth_minmax
        #     loss_data['depth_minmax'] = loss_depth_minmax.mean().item()
        return loss, loss_data

    def train(self, batch):
        self.model.zero_grad()
        pred = self.forward(batch['RGB_in'] / 255.)
        gt = {
            "depth": batch['depth'],
            "silhou": batch["silhou"],
            "normal": batch["normal"]
        }
        loss, loss_data = self.compute_loss(pred, gt)
        loss.backward()
        self.optimizer.step()
        batch_size = len(batch)
        batch_log = {'size': batch_size, **loss_data}
        return batch_log


if __name__ == "__main__":
    deal_dataset = DealDataset()
    train_loader = DataLoader(dataset=deal_dataset,
                              batch_size=32,
                              shuffle=True)
    model = Model()
    print(model)
    model.apply(weights_init)
    model = model.to(cuda_5)

    for i, data in enumerate(train_loader):
        inputs_RGB = Variable(data['RGB'].to(cuda_5))
        inputs_depth = Variable(data['depth'].to(cuda_5))
        inputs_silhou = Variable(data['silhou'].to(cuda_5))
        inputs_normal = Variable(data['normal'].to(cuda_5))

        inputs = {
            "RGB_in": inputs_RGB,
            "depth": inputs_depth,
            "silhou": inputs_silhou,
            "normal": inputs_normal
        }

        batch_log = model.train(inputs)
        print(batch_log)

    # eval by pre-trained model
    # PATH = r'/mnt/zhengwen/model_synthesis/MarrNetDataModel/downloads/models/marrnet1_with_minmax.pt'
    # model.load_state_dict(torch.load(PATH)["nets"][0])
    #
    #
    # optimizer = optim.Adam(
    #     model.parameters(),
    # )
    # model.eval()
    # model.cuda()
    # input_image = Image.open("/mnt/zhengwen/model_synthesis/shapeNetCore/ShapeNetCore.v1/02691156/fff513f407e00e85a9ced22d91ad7027/rendered_9.png")
    # input_image = np.asarray(input_image)[None, :, :, :3]
    # input_image = input_image.transpose((0, 3, 1, 2)).astype("float32") / 255.
    # with torch.no_grad():
    #     output = model(torch.from_numpy(input_image).cuda())
    # normal = postprocess(output['normal'].detach().cpu(), SCALE_25D)
    # depth = postprocess(output['depth'].detach().cpu(), SCALE_25D)
    # silhou = postprocess(output['silhou'].detach().cpu(), SCALE_25D)
    # normal = normal.numpy()
    #
    # np.save("normal.npy", normal)
    # np.save("depth.npy", depth)
    # np.save("silhou.npy", silhou)
