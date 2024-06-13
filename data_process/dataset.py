import numpy as np
from PIL import Image
from torchvision import transforms


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_dataset":
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # 随机颜色抖动
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    return transform


def My_dataset(config):
    dsets = dict()
    data_config = config["data_list"]

    for data_set in ["train_dataset", "test_dataset", "database_dataset"]:
        dsets[data_set] = ImageList(config["data_path"], open(data_config[data_set]).readlines(),
                                    transform=image_transform(config["resize_size"], config["crop_size"], data_set))

        print(data_set, len(dsets[data_set]))

    return dsets["train_dataset"], dsets["test_dataset"], dsets["database_dataset"], \
        len(dsets["train_dataset"]), len(dsets["test_dataset"]), len(dsets["database_dataset"])
