import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
import io
import numpy as np


def get_model_instance_segmentation(num_classes, pretrained_path):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None)

    model.load_state_dict(torch.load(pretrained_path))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model_instance_segmentation_mobile(num_classes, pretrained_path):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights='DEFAULT')

    #model.load_state_dict(torch.load(pretrained_path))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_model_instance_segmentation_mobile(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights='DEFAULT')

    #model.load_state_dict(torch.load(pretrained_path))

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class TangleAIDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, bucket, prefix, target_dict, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.bucket = bucket
        self.transform = transform
        self.prefix = prefix
        self.target_dict = target_dict

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        folder = self.df.stain[idx]
        blob_name = f"{self.prefix}/{folder}/{self.df.loc[idx, 'image_name']}"
        blob = self.bucket.blob(blob_name)
        blob = blob.download_as_string()
        img = Image.open(io.BytesIO(blob)).convert('RGB')

        image_id = torch.as_tensor(self.df['index'][idx])
        if 'none' in self.df['labels'][idx]:
            labels = torch.as_tensor([])
            bboxes = torch.as_tensor(np.empty((0, 4)))
        else:
            bboxes = torch.as_tensor(np.array(self.df.bboxes[idx], dtype=float))
            labels = torch.as_tensor([self.target_dict[l] for l in self.df.labels[idx]])

        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        """
        if self.transform:
            img, target = self.transform(img,target)
        """
        if self.transform is not None:
            transformed = self.transform(
                image=np.array(img),
                bboxes=target['boxes'],
                class_labels=labels
            )
            img = transformed["image"]
            target['boxes'] = torch.tensor(np.array(transformed['bboxes']))
            target['labels'] = torch.tensor(transformed['class_labels'])
        if 'none' in self.df['labels'][idx]:
            target['labels'] = torch.as_tensor([], dtype=torch.int64)
            target['boxes'] = torch.as_tensor(np.empty((0, 4)))

        return img, target
