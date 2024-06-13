import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from model import HashingModel
from utils import SoftSort, SortedNCELoss, data_supply
from evaluate import evalModel
from data_process.dataloader import my_dataloader, config_dataset
import time
import sys
import os
import warnings

warnings.filterwarnings("ignore")


def get_config():
    configs = {
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        "optimizer": {"type": optim.Adam, "optim_params": {"lr": 1e-5}},
        "info": "NSH",
        "dataset": "mirflickr",
        "net": HashingModel,
        "resize_size": 224,
        "crop_size": 224,
        "batch_size": 16,
        "epoch": 200,
        "bit_list": [16, 32, 64, 128],
        "positive_num": 2,
        "tau": 0.1,
        "num_workers": 4,
        "test_map": 1,
        "logs_path": "results",
    }
    configs = config_dataset(configs)
    return configs


def train_nsh(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = my_dataloader(config)
    config["num_train"] = num_train

    # 定义
    model = config["net"](bit).to(device)
    optimizer = config["optimizer"]["type"](model.parameters(), **(config["optimizer"]["optim_params"]))
    softsort = SoftSort(bit)
    sorted_nce_loss = SortedNCELoss(config["positive_num"], config["tau"], config["batch_size"])

    best_map = 0

    # log文件
    logs_path = os.path.join(config['logs_path'], f"{config['dataset']}_{bit}bit")
    os.makedirs(logs_path, exist_ok=True)
    config['logs_path'] = logs_path
    log_filename = "logs_{}_{}_{}.txt".format(config['info'], config['dataset'], bit)
    log_filepath = os.path.join(config['logs_path'], log_filename)
    if os.path.exists(log_filepath):
        os.remove(log_filepath)
    log_file = open(log_filepath, 'a')
    for key, value in config.items():
        log_file.write(f"{key}: {value}\n")
    log_file.write("\n\n")

    start_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print(f"\033[31m model:{config['info']} start_time:[{start_time}] bit:{bit}, dataset:{config['dataset']} \033[0m")
    print("\033[31m--------------Training begins--------------\033[0m")

    for epoch in range(config["epoch"]):
        model.train()
        epoch_loss = 0
        data_loader = tqdm(train_loader, file=sys.stdout)
        for step, batch in enumerate(data_loader):
            images, labels, indices = batch
            # 最后一批若不等于batch大小，复制补充
            if images.shape[0] < config["batch_size"]:
                images = data_supply(batch, images, config["batch_size"])

            images = images.to(device)
            optimizer.zero_grad()

            # [b1, z1], [b2, z2] = f_theta(batch1), f_theta(batch2)
            [b1, z1], [b2, z2] = model(images), model(images)

            # s = torch.matmul(b1, b2.T) / 2 / d_l + 0.5  # 1 - Hamming/d_l
            s = torch.matmul(b1, b2.T) / (2 * bit) + 0.5
            # p = softsort(-s)  # [n, n, n]，参数论文中为-s,在此函数中应为s
            p = softsort(s)
            # e = torch.einsum('nnn,nd->nnd', p, z1)
            e = torch.einsum('ijk, kl-> ijl', p, z1)

            # labels = onehot(torch.zeros(n), n - m + 1)
            length = config["batch_size"] - config["positive_num"] + 1
            labels_onehot = np.zeros((config["batch_size"], length), dtype=float)
            labels_onehot[:, 0] = 1
            labels_onehot = torch.from_numpy(labels_onehot).to(device)

            # cos = torch.einsum('nnd,nd->nn', e, z2)，计算余弦相似度矩阵cos
            norm_e_rows = torch.norm(e, dim=2)
            norm_z_rows = torch.norm(z2, dim=1)
            cos = torch.einsum('ijk,ik->ij', e, z2) / (norm_e_rows * norm_z_rows[:, None])

            # loss = SortNCELoss + quantization_loss
            loss = sorted_nce_loss(cos, labels_onehot)
            h1 = torch.einsum('ijk, kl-> ijl', p, b1)
            h2 = torch.einsum('ijk, kl-> ijl', p, b2)
            # 梯度不参与更新
            b1 = b1.detach()
            b2 = b2.detach()
            quantization_loss = ((torch.norm(b1 - h1) ** 2) + (torch.norm(b2 - h2) ** 2)) / (2 * config["batch_size"])
            loss += quantization_loss

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()

            data_loader.desc = f"[train epoch {epoch}] loss: {epoch_loss / (step + 1):.5f}"

        epoch_loss /= len(train_loader)
        log_file.write(f"Epoch {epoch + 1}: Train Loss = {epoch_loss}\n")

        if (epoch + 1) % config["test_map"] == 0:
            best_map = evalModel(test_loader, dataset_loader, model, best_map, bit, config, epoch, log_file)

    print("\033[31m--------------Training ended--------------\033[0m")
    end_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print(f"\033[31m model:{config['info']} end_time:[{end_time}] bit:{bit}, dataset:{config['dataset']} \033[0m")


if __name__ == "__main__":
    config = get_config()
    for bit in config["bit_list"]:
        train_nsh(config, bit)
        config = get_config()
