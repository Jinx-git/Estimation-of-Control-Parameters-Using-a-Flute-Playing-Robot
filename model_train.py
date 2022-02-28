import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from mlp_mixer_pytorch import MLPMixer
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import glob
import os
import itertools
from tqdm import tqdm
from Model.Dataset import Dataset, HumanDataset
from argparse import ArgumentParser

def get_option(batch, epoch, rank_ratio, lr):
    argparser = ArgumentParser()
    argparser.add_argument('n')
    argparser.add_argument('-b', '--batch', type=int,
                           default=batch,
                           help='Specify size of batch')
    argparser.add_argument('-e', '--epoch', type=int,
                           default=epoch,
                           help='Specify number of epoch')
    argparser.add_argument('-r', '--rankTrainRatio', type=float,
                           default=rank_ratio,
                           help='Coefficients for ranking learning')
    argparser.add_argument('-lr', '--learningRate', type=float,
                           default=lr,
                           help='Learning Rate')
    return argparser.parse_args()


def loss_f(x, y, t):
    o = x-y
    loss = (-t * o + F.softplus(o)).mean()
    return loss


def main():
    test_size = 0.2

    args = get_option(128, 100, 0.1, 0.00005)

    model_name = args.n
    BATCH_SIZE = args.batch
    EPOCH = args.epoch
    rank_ratio = args.rankTrainRatio
    LEARNING_RATE = args.learningRate

    CPU_NUM= os.cpu_count()
    print("use " + str(CPU_NUM) + " cpus")
    save_model_dir = os.path.join("./Model/saves/" + model_name)

    if not os.path.exists(save_model_dir):
        os.mkdir(save_model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_robot = []
    val_robot = []
    train_human = []
    val_human = []

    robot_dirs = glob.glob("./datasets/train/spectrogram/robot_performance/*/*")
    for robot_dir in robot_dirs:
        robot_datas = glob.glob(robot_dir + "/*")
        tmp_train, tmp_val = train_test_split(robot_datas, test_size=test_size)
        train_robot.append(tmp_train)
        val_robot.append(tmp_val)
    human_dirs = glob.glob("./datasets/train/spectrogram/human_performance/*/*")
    for human_dir in human_dirs:
        human_datas = glob.glob(human_dir + "/*")
        tmp_train, tmp_val = train_test_split(human_datas, test_size=test_size)
        train_human.append(tmp_train)
        val_human.append(tmp_val)

    train_robot = list(itertools.chain.from_iterable(train_robot))
    val_robot = list(itertools.chain.from_iterable(val_robot))
    train_human = list(itertools.chain.from_iterable(train_human))
    val_human = list(itertools.chain.from_iterable(val_human))

    train_robot_dataset = Dataset(file_list=train_robot)
    print("train robot data : ", len(train_robot_dataset))
    train_robot_loader = DataLoader(train_robot_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=CPU_NUM, pin_memory=True)

    val_robot_dataset = Dataset(file_list=val_robot)
    print("validation robot data : ", len(val_robot_dataset))
    val_robot_loader = DataLoader(val_robot_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=CPU_NUM, pin_memory=True)

    robot_dataloaders = {"train": train_robot_loader, "val": val_robot_loader}

    train_human_dataset = HumanDataset(file_list=train_human)
    print("train human data : ", len(train_human_dataset))
    train_human_loader = DataLoader(train_human_dataset, batch_size=BATCH_SIZE,
                                    shuffle=True, num_workers=CPU_NUM, pin_memory=True)

    val_human_dataset = HumanDataset(file_list=val_human)
    print("human validation data : ", len(val_human_dataset))
    val_human_loader = DataLoader(val_human_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=CPU_NUM, pin_memory=True)

    human_dataloaders = {"train": train_human_loader, "val": val_human_loader}
    net_angle = MLPMixer(image_size=128,
                         patch_size=16,
                         dim=512,
                         depth=3,
                         num_classes=1,
                         channels=1)
    net_flow = MLPMixer(image_size=128,
                        patch_size=16,
                        dim=512,
                        depth=3,
                        num_classes=1,
                        channels=1)
    net_angle = net_angle.to(device)
    net_flow = net_flow.to(device)
    optimizer_angle = optim.Adam(net_angle.parameters(), lr=LEARNING_RATE)
    optimizer_flow = optim.Adam(net_flow.parameters(), lr=LEARNING_RATE)

    loss_train_angle = []
    loss_val_angle = []
    loss_train_flow = []
    loss_val_flow = []

    with tqdm(total=EPOCH, unit="epoch") as pbar:
        for epoch in range(EPOCH):
            pbar.set_description(f"Epoch[{epoch + 1}/{EPOCH}]")
            for phase in robot_dataloaders.keys():
                sum_loss_angle = 0
                sum_loss_flow = 0
                data_num_angle = 0
                data_num_flow = 0
                if phase == "train":
                    net_angle.train()
                    net_flow.train()
                else:
                    net_angle.eval()
                    net_flow.eval()
                if (epoch == 0) and (phase == "train"):
                    loss_train_angle.append(0.0)
                    loss_train_flow.append(0.0)
                    pbar.update(1)
                    continue
                for ((inputs, angles, flows), (human_inputs1, human_inputs2, human_angles, human_flows)) \
                        in zip(robot_dataloaders[phase], human_dataloaders[phase]):
                    optimizer_angle.zero_grad()
                    optimizer_flow.zero_grad()
                    inputs, angles, flows = inputs.to(device), angles.to(device), flows.to(device)
                    human_inputs1, human_inputs2 = human_inputs1.to(device), human_inputs2.to(device)
                    human_angles, human_flows = human_angles.to(device), human_flows.to(device)

                    outputs_robot_angle = torch.sigmoid(net_angle(inputs))
                    outputs_human1_angle = net_angle(human_inputs1)
                    outputs_human2_angle = net_angle(human_inputs2)

                    outputs_robot_flow = torch.sigmoid(net_flow(inputs))
                    outputs_human1_flow = net_flow(human_inputs1)
                    outputs_human2_flow = net_flow(human_inputs2)

                    loss_robot_angle = nn.functional.mse_loss(outputs_robot_angle[:, 0], angles.view(-1))
                    loss_human_angle = loss_f(outputs_human1_angle, outputs_human2_angle, human_angles)
                    loss_robot_flow = nn.functional.mse_loss(outputs_robot_flow[:, 0], flows.view(-1))
                    loss_human_flow = loss_f(outputs_human1_flow, outputs_human2_flow, human_flows)

                    loss_angle = loss_robot_angle + rank_ratio * loss_human_angle
                    loss_flow = loss_robot_flow + rank_ratio * loss_human_flow
                    # loss_robot.backward()

                    loss_angle.backward()
                    optimizer_angle.step()
                    loss_flow.backward()
                    optimizer_flow.step()

                    sum_loss_angle += loss_angle.item() * len(angles)
                    sum_loss_flow += loss_flow.item() * len(flows)
                    data_num_angle += len(angles)
                    data_num_flow += len(flows)

                tmp_loss_angle = sum_loss_angle / data_num_angle
                tmp_loss_flow = sum_loss_flow / data_num_flow

                if phase is "train":
                    loss_train_angle.append(tmp_loss_angle)
                    loss_train_flow.append(tmp_loss_flow)
                else:
                    loss_val_angle.append(tmp_loss_angle)
                    loss_val_flow.append(tmp_loss_flow)

            pbar.set_postfix({'train angle': loss_train_angle[-1],
                              'val angle': loss_val_angle[-1],
                              'train flow': loss_train_flow[-1],
                              'val flow': loss_val_flow[-1]})
            pbar.update(1)
            if epoch % 10 == 0:
                torch.save(net_angle.state_dict(), save_model_dir + "/angle_{}.pth".format(epoch))
                torch.save(net_flow.state_dict(), save_model_dir + "/flow_{}.pth".format(epoch))

    torch.save(net_angle.state_dict(), save_model_dir + "/angle.pth")
    torch.save(net_flow.state_dict(), save_model_dir + "/flow.pth")


if __name__ == "__main__":
    main()

