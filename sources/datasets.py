import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")

from tqdm.notebook import tqdm

# Библиотеки для обучения
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

# Библиотеки для обработки изображений
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import scipy.io

def prepare_class_extractor(cls=1):
    def class_extractor(mask, **kwargs):
        return (mask == cls).astype(int)

    return class_extractor


COCO_INPUT_SIZE = 320
CITYSCAPES_INPUT_SIZE = (256, 512)

coco_transform = A.Compose([
    A.Lambda(mask=prepare_class_extractor(cls=1)),
    A.LongestMaxSize(COCO_INPUT_SIZE, cv2.INTER_LINEAR),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.PadIfNeeded(COCO_INPUT_SIZE, COCO_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
    ToTensorV2(),
])


cs_transform = A.Compose([
    A.Lambda(mask=prepare_class_extractor(cls=1)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


#!g1.1
class CocoStuffDataset(Dataset):
    def __init__(self, image_path, mask_path, filenames, transform=None, preprocess=False):
        """
        Args:
            image_path: путь до изображений
            mask_path: путь до масок
            filenames: список изображений
            transform: трансформация для изображений.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.filenames = filenames
        self.transform = transform
        self.preprocess = preprocess

        if preprocess:
            self.preprocessed = []
            for filename in tqdm(filenames):
                self.preprocessed.append(
                    self.transform(
                        image=cv2.cvtColor(
                            cv2.imread(os.path.join(self.image_path, filename + ".jpg")),
                            cv2.COLOR_BGR2RGB,
                        ),
                        mask=scipy.io.loadmat(
                            os.path.join(self.mask_path, filename + ".mat"),
                        )['S'],
                    )
                )

    def __len__(self):
        return len(self.filenames)

    def get_preprocessed_image(self, idx):
        return self.preprocessed[idx]

    def get_processed_image(self, idx):
        return self.transform(
            image=cv2.cvtColor(
                cv2.imread(os.path.join(self.image_path, self.filenames[idx] + ".jpg")),
                cv2.COLOR_BGR2RGB,
            ),
            mask=scipy.io.loadmat(
                os.path.join(self.mask_path, self.filenames[idx] + ".mat"),
            )['S'],
        )

    def __getitem__(self, idx):
        if self.preprocess:
            return self.get_preprocessed_image(idx)
        return self.get_processed_image(idx)
    

def setup_cocostuff_dataloaders(dataset_split, gt_path, image_path, batch_size,
                      k=8, transform=coco_transform, preprocess=False):
    train = dataset_split[(dataset_split['split'] == "train") & (dataset_split['group'] < k)].filenames.tolist()
    valid = dataset_split[(dataset_split['split'] == "valid") & (dataset_split['group'] < k)].filenames.tolist()
    test = dataset_split[(dataset_split['split'] == "test") & (dataset_split['group'] < k)].filenames.tolist()

    train_dataset = CocoStuffDataset(
        image_path,
        gt_path,
        train,
        transform=transform,
        preprocess=preprocess,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )


    valid_dataset = CocoStuffDataset(
        image_path,
        gt_path,
        valid,
        transform=transform,
        preprocess=preprocess,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )


    test_dataset = CocoStuffDataset(
        image_path,
        gt_path,
        test,
        transform=transform,
        preprocess=preprocess,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    return {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
    }


class CocoCarsDataset(Dataset):
    def __init__(self, image_path, mask_path, filenames, transform=None, preprocess=False):
        """
        Args:
            image_path: путь до изображений
            mask_path: путь до масок
            filenames: список изображений
            transform: трансформация для изображений.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.filenames = filenames
        self.transform = transform
        self.preprocess = preprocess

        if preprocess:
            self.preprocessed = []
            for filename in tqdm(filenames):
                self.preprocessed.append(
                    self.transform(
                        image=cv2.cvtColor(
                            cv2.imread(os.path.join(self.image_path, filename)),
                            cv2.COLOR_BGR2RGB,
                        ),
                        mask=cv2.imread(
                            os.path.join(self.mask_path, filename),
                            cv2.IMREAD_GRAYSCALE,
                        ),
                    )
                )

    def __len__(self):
        return len(self.filenames)

    def get_preprocessed_image(self, idx):
        return self.preprocessed[idx]

    def get_processed_image(self, idx):
        return self.transform(
            image=cv2.cvtColor(
                cv2.imread(os.path.join(self.image_path, self.filenames[idx])),
                cv2.COLOR_BGR2RGB,
            ),
            mask=cv2.imread(
                os.path.join(self.mask_path, self.filenames[idx]),
                cv2.IMREAD_GRAYSCALE,
            ),
        )

    def __getitem__(self, idx):
        if self.preprocess:
            return self.get_preprocessed_image(idx)
        return self.get_processed_image(idx)


def setup_cococars_dataloaders(gt_path, image_path, batch_size, train, test, valid,
                           transform=coco_transform, preprocess=False):
    train_dataset = CocoCarsDataset(
        image_path,
        gt_path,
        train,
        transform=transform,
        preprocess=preprocess,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )


    valid_dataset = CocoCarsDataset(
        image_path,
        gt_path,
        valid,
        transform=transform,
        preprocess=preprocess,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )


    test_dataset = CocoCarsDataset(
        image_path,
        gt_path,
        test,
        transform=transform,
        preprocess=preprocess,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    return {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
    }


class CityscapesDataset(Dataset):
    def __init__(self, image_path, mask_path, filenames, transform=None, preprocess=False):
        """
        Args:
            image_path: путь до изображений
            mask_path: путь до масок
            filenames: список изображений
            transform: трансформация для изображений.
        """
        self.image_path = image_path
        self.mask_path = mask_path
        self.filenames = filenames
        self.transform = transform
        self.preprocess = preprocess

        if preprocess:
            self.preprocessed = []
            for filename in tqdm(filenames):
                self.preprocessed.append(
                    self.transform(
                        image=cv2.cvtColor(
                            cv2.imread(os.path.join(self.image_path, filename)),
                            cv2.COLOR_BGR2RGB,
                        ),
                        mask=cv2.imread(
                            os.path.join(self.mask_path, filename),
                            cv2.IMREAD_GRAYSCALE,
                        ),
                    )
                )

    def __len__(self):
        return len(self.filenames)

    def get_preprocessed_image(self, idx):
        return self.preprocessed[idx]

    def get_processed_image(self, idx):
        return self.transform(
            image=cv2.cvtColor(
                cv2.imread(os.path.join(self.image_path, self.filenames[idx])),
                cv2.COLOR_BGR2RGB,
            ),
            mask=cv2.imread(
                os.path.join(self.mask_path, self.filenames[idx]),
                cv2.IMREAD_GRAYSCALE,
            ),
        )

    def __getitem__(self, idx):
        if self.preprocess:
            return self.get_preprocessed_image(idx)
        return self.get_processed_image(idx)


def setup_cityscapes_dataloaders(gt_path, image_path, batch_size,
                                 transform, preprocess=False,
                                 valid_cities=['bochum', 'ulm', 'krefeld', 'monchengladbach', 'zurich']):
    train_gt_path = os.path.join(gt_path, "train")
    valid_gt_path = os.path.join(gt_path, "train")
    test_gt_path = os.path.join(gt_path, "val")
    
    train_image_path = os.path.join(image_path, "train")
    valid_image_path = os.path.join(image_path, "train")
    test_image_path = os.path.join(image_path, "val")
    
    train = sorted(list(filter(lambda x: x.split("_")[0] not in valid_cities, os.listdir(train_image_path))))
    valid = sorted(list(filter(lambda x: x.split("_")[0] in valid_cities, os.listdir(train_image_path))))
    test = sorted(os.listdir(test_image_path))
    
    print(len(train), len(valid), len(test))

    train_dataset = CityscapesDataset(
        train_image_path,
        train_gt_path,
        train,
        transform=transform,
        preprocess=preprocess,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2,
    )


    valid_dataset = CityscapesDataset(
        valid_image_path,
        valid_gt_path,
        valid,
        transform=transform,
        preprocess=preprocess,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )


    test_dataset = CityscapesDataset(
        test_image_path,
        test_gt_path,
        test,
        transform=transform,
        preprocess=preprocess,
    )

    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
    )

    return {
        "train": train_dataloader,
        "valid": valid_dataloader,
        "test": test_dataloader,
    }