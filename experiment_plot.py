import os.path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import glob
# from Model import AlexNet, MobileNet, ResNet

from Model import Dataset
from mlp_mixer_pytorch import MLPMixer
import matplotlib.pyplot as plt

participant = sys.argv[1]


def main(participant):
    modes = [
        "angle",
        "flow"
    ]
    participant_dir = "./datasets/experiment/spectrogram/{}".format(participant)
    save_fig_dir = "./result/{}".format(participant)
    if not os.path.exists(save_fig_dir):
        os.mkdir(save_fig_dir)
    models = [
        "robot",
        "human",
        "both"
    ]
    device = torch.device("cpu")

    net_flow = MLPMixer(
        image_size=128,
        patch_size=16,
        dim=512,
        depth=3,
        num_classes=1,
        channels=1
    )
    net_angle = MLPMixer(
        image_size=128,
        patch_size=16,
        dim=512,
        depth=3,
        num_classes=1,
        channels=1
    )
    net_angle.to(device)
    net_flow.to(device)
    net_angle.eval()
    net_flow.eval()

    for mode in modes:
        n = 0
        for model in models:
            net_angle.load_state_dict(torch.load("./Model/saves/{}/angle.pth".format(model),
                                                 map_location=torch.device("cpu")))
            net_flow.load_state_dict(torch.load("./Model/saves/{}/flow.pth".format(model),
                                                map_location=torch.device("cpu")))
            val_files = glob.glob("{}/{}/*/*".format(participant_dir, mode))
            val_dataset = Dataset.ExperimentDataset(file_list=val_files)
            val_loader = DataLoader(val_dataset,
                                    batch_size=1,
                                    shuffle=False,
                                    num_workers=0)
            if mode == "angle":
                datas_angle = {"back": [],
                               "center": [],
                               "front": []}
                datas_flow = {"back": [],
                              "center": [],
                              "front": []}
            else:
                datas_angle = {"p": [],
                               "f": []}
                datas_flow = {"p": [],
                              "f": []}

            fig, ax = plt.subplots()

            for inputs,  labels, filename in val_loader:
                inputs = inputs.to(device)
                out_angle = torch.sigmoid(net_angle(inputs))
                out_flow = torch.sigmoid(net_flow(inputs))
                if mode == "angle":
                    datas_angle[labels[0]].append(out_angle[0, 0].item())
                    datas_flow[labels[0]].append((0.55 - 0.45) * np.random.rand() + 0.45)
                else:
                    datas_angle[labels[0]].append((0.55 - 0.45) * np.random.rand() + 0.45)
                    datas_flow[labels[0]].append(out_flow[0, 0].item())

            if mode == "angle":
                ax.scatter(datas_angle["front"], datas_flow["front"], c="red", marker="^", label="FRONT")
                ax.scatter(datas_angle["center"], datas_flow["center"], c="green", marker="o", label="CENTER")
                ax.scatter(datas_angle["back"], datas_flow["back"], c="blue", marker="x", label="BACK")
            else:
                ax.scatter(datas_angle["f"], datas_flow["f"], c="red", marker="^", label="F")
                ax.scatter(datas_angle["p"], datas_flow["p"], c="blue", marker="x", label="P")
            ax.set_xlabel("angle")
            ax.set_ylabel("Flow")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.minorticks_on()
            if mode == "angle":
                ax.grid(axis="x", color="black", which="major")
                ax.grid(axis="x", color="gray", linestyle=":", which="minor")
            else:
                ax.grid(axis="y", color="black", which="major")
                ax.grid(axis="y", color="gray", linestyle=":", which="minor")
            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)
            ax.set_aspect('equal')
            ax.set_title(models[n])
            fig.savefig("{}/{}_{}.png".format(save_fig_dir, mode, os.path.basename(model)), bbox_inches='tight', pad_inches=0)
            plt.close()
            n += 1


if __name__ == "__main__":
    main(participant)
