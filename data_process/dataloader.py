import torch
from .cifar_dataset import MyCIFAR10_dataset
from .dataset import My_dataset


def my_dataloader(config):
    if "cifar10" in config["dataset"]:
        train_dataset, test_dataset, database_dataset, num_train, num_test, num_database = MyCIFAR10_dataset(config)
    else:
        train_dataset, test_dataset, database_dataset, num_train, num_test, num_database = My_dataset(config)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=config['num_workers'])

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              num_workers=config['num_workers'])

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=config['batch_size'],
                                                  shuffle=False,
                                                  num_workers=config['num_workers'])
    return train_loader, test_loader, database_loader, num_train, num_test, num_database


def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = 1000
        config["n_class"] = 10
    elif config["dataset"] == "nuswide_21":
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_10":
        config["topK"] = 5000
        config["n_class"] = 10
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "mirflickr":
        config["topK"] = 5000
        config["n_class"] = 24

    if "cifar" in config["dataset"]:
        config["data_path"] = "./dataset/cifar/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "nuswide_10":
        config["data_path"] = "./dataset/NUS-WIDE/"
    if config["dataset"] == "coco":
        config["data_path"] = "./dataset/coco/"
    if config["dataset"] == "mirflickr":
        config["data_path"] = "./dataset/flickr25k/mirflickr/"
    config["data_list"] = {
        "train_dataset": "./data/" + config["dataset"] + "/train.txt",
        "test_dataset": "./data/" + config["dataset"] + "/test.txt",
        "database_dataset": "./data/" + config["dataset"] + "/database.txt"}
    return config


