import torch
import torchvision.transforms
from torch.utils import data
import os
import numpy as np
import random


class Dataset(data.Dataset):
    def __init__(self, file_list, angle=(0, 0), flow=(0.0, 0.0)):
        self.file_list = file_list
        self.angle = angle
        self.flow = flow

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # ラベルの取り出し
        flow_path, _ = os.path.split(self.file_list[index])
        flow = float(os.path.basename(flow_path))
        angle_path, _ = os.path.split(flow_path)
        angle = float(os.path.basename(angle_path))
        # print(flow, angle)

        # ラベルの正規化、テンソル化
        angle = np.array([(angle-self.angle[0]) / (self.angle[1] - self.angle[0])])
        flow = np.array([(flow - self.flow[0]) / (self.flow[1] - self.flow[0])])
        angle = torch.from_numpy(angle.astype(np.float32)).clone()
        flow = torch.from_numpy(flow.astype(np.float32)).clone()
        # print(flow, angle)

        # numpyデータのロード
        np_data = np.load(self.file_list[index])
        # ランダムで開始位置を指定
        max_len = np_data.shape[1] - np_data.shape[0]
        rand_index = random.randrange(max_len+1)
        # (128,128)に切り出し
        np_data = np_data[:, rand_index:rand_index+128]
        # データのテンソル化、リサイズ
        tensor_data = torchvision.transforms.ToTensor()(np_data)

        return tensor_data, angle, flow


class HumanDataset(data.Dataset):
    def __init__(self, file_list, human_dict=None):
        self.file_list = file_list
        self.human_dict = human_dict
        self.angle = ["DOWN", "MID", "UP"]
        self.flow = ["P", "M", "F"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # ラベルの取り出し
        flow_path, _ = os.path.split(self.file_list[index])
        flow = os.path.basename(flow_path)
        angle_path, _ = os.path.split(flow_path)
        angle = os.path.basename(angle_path)
        human_name_path, _ = os.path.split(angle_path)
        human_name = os.path.basename(human_name_path)

        # numpyデータのロード
        np_data = np.load(self.file_list[index])
        tensor_data = torchvision.transforms.ToTensor()(np_data)

        datalen = len(self.human_dict[human_name])

        t_rand = random.randrange(datalen)
        flow_path_t, _ = os.path.split(self.human_dict[human_name][t_rand])
        flow_t = os.path.basename(flow_path_t)
        angle_path_t, _ = os.path.split(flow_path_t)
        angle_t = os.path.basename(angle_path_t)

        # numpyデータのロード
        np_data_t = np.load(self.human_dict[human_name][t_rand])
        # データのテンソル化、リサイズ
        tensor_data_t = torchvision.transforms.ToTensor()(np_data_t)

        flow_class = self.flow.index(flow) - self.flow.index(flow_t)
        angle_class = self.angle.index(angle) - self.angle.index(angle_t)

        if angle_class > 0:
            angle_class = 1
        elif angle_class < 0:
            angle_class = 0
        else:
            angle_class = 0.5

        if flow_class > 0:
            flow_class = 1
        elif flow_class < 0:
            flow_class = 0
        else:
            flow_class = 0.5

        return tensor_data, tensor_data_t, angle_class, flow_class


class ExperimentDataset(data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        label_path, filename = os.path.split(self.file_list[index])
        label = os.path.basename(label_path)

        np_data = np.load(self.file_list[index])
        tensor_data = torchvision.transforms.ToTensor()(np_data)
        return tensor_data, label, filename

