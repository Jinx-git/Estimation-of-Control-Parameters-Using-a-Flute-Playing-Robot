import torch
import torchvision.transforms
from torch.utils import data
import os
import numpy as np
import random


class Dataset(data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.angle = (1400, 1680)
        self.flow = (0.570, 0.620)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        flow_path, _ = os.path.split(self.file_list[index])
        flow = float(os.path.basename(flow_path))
        angle_path, _ = os.path.split(flow_path)
        angle = float(os.path.basename(angle_path))

        angle = np.array([(angle-self.angle[0]) / (self.angle[1] - self.angle[0])])
        flow = np.array([(flow - self.flow[0]) / (self.flow[1] - self.flow[0])])
        angle = torch.from_numpy(angle.astype(np.float32)).clone()
        flow = torch.from_numpy(flow.astype(np.float32)).clone()

        np_data = np.load(self.file_list[index])
        tensor_data = torchvision.transforms.ToTensor()(np_data)

        return tensor_data, angle, flow


class HumanDataset(data.Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.angle = ["back", "front"]
        self.flow = ["p", "f"]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        flow_path, _ = os.path.split(self.file_list[index])
        flow = os.path.basename(flow_path)
        angle_path, _ = os.path.split(flow_path)
        angle = os.path.basename(angle_path)
        human_name_path, _ = os.path.split(angle_path)

        np_data = np.load(self.file_list[index])
        tensor_data = torchvision.transforms.ToTensor()(np_data)

        datalen = len(self.file_list)

        t_rand = random.randrange(datalen)
        flow_path_t, _ = os.path.split(self.file_list[t_rand])
        flow_t = os.path.basename(flow_path_t)
        angle_path_t, _ = os.path.split(flow_path_t)
        angle_t = os.path.basename(angle_path_t)

        np_data_t = np.load(self.file_list[t_rand])
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

