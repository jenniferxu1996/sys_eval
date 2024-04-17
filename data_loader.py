from sklearn.preprocessing import label_binarize
from torch.utils import data
from PIL import Image
import model_infer.utils_mpii_cooking as utils_mpii_cooking
import os
import pandas as pd
import torch
import numpy as np


class SingleFrameDataloader(data.Dataset):
    def __init__(self, args, transform=None):
        self.dataset = args.dataset
        self.file_path = f'{args.data_dir_path}/{args.dataset}/frame_16'
        self.args = args
        self.sampled_frame = args.sampled_frame
        self.mode = args.mode
        if self.dataset == 'mpii_cooking':
            if self.mode == 'validation':
                self.all_videos = [each_video for each_video in os.listdir(self.file_path) if
                                   each_video.split('_')[0] in utils_mpii_cooking.VALID + utils_mpii_cooking.TEST]
            else:
                self.all_videos = [each_video for each_video in os.listdir(self.file_path) if
                                   each_video.split('_')[0] not in utils_mpii_cooking.VALID + utils_mpii_cooking.TEST]
            self.label_map = utils_mpii_cooking.label_mapping(args)
            self.annotation_df = pd.read_csv(f"{args.data_dir_path}/mpii_cooking/cooking.csv")
        elif self.dataset == 'epic_kitchen':
            self.all_videos = os.listdir(os.path.join(self.file_path, self.mode))
            self.annotation_df = pd.read_csv(f"{args.data_dir_path}/epic_kitchen/EPIC_100_{self.mode}.csv")
        self.transform = transform

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        return self.all_videos[index], self.load_frame_and_target(self.all_videos[index])

    def load_frame_and_target(self, each_video):
        if len(os.listdir(os.path.join(self.file_path, each_video))) == 16:
            sampled_frame_path = os.path.join(self.file_path, each_video, f"{self.sampled_frame}.jpg")
        else:
            sampled_frame_path = os.path.join(self.file_path, each_video, f"{int(len(os.listdir(os.path.join(self.file_path, each_video)))/2)}.jpg")
        input_img = self.transform(Image.open(sampled_frame_path))
        target = None
        if self.dataset == 'mpii_cooking':
            selected_annotation = self.annotation_df.loc[self.annotation_df['video_id'] == each_video]
            target = self.label_map[selected_annotation['label'].values[0]]
        elif self.dataset == 'epic_kitchen':
            selected_annotation = self.annotation_df.loc[self.annotation_df['narration_id'] == each_video]
            target = selected_annotation['verb_class'].values[0]
        target = np.asarray(label_binarize([target], classes=[i for i in range(self.args.num_class)])).ravel()
        return torch.Tensor(input_img), torch.Tensor(target)


class SeqDataloader(data.Dataset):
    def __init__(self, args, transform=None):
        self.dataset = args.dataset
        self.file_path = f'{args.data_dir_path}/{args.dataset}/frame_16'
        self.args = args
        self.sampled_frame = args.sampled_frame
        self.mode = args.mode
        if self.dataset == 'mpii_cooking':
            if self.mode == 'validation':
                self.all_videos = [each_video for each_video in os.listdir(self.file_path) if
                                   each_video.split('_')[0] in utils_mpii_cooking.VALID + utils_mpii_cooking.TEST]
            else:
                self.all_videos = [each_video for each_video in os.listdir(self.file_path) if
                                   each_video.split('_')[0] not in utils_mpii_cooking.VALID + utils_mpii_cooking.TEST]
            self.label_map = utils_mpii_cooking.label_mapping(args)
            self.annotation_df = pd.read_csv(f"{args.data_dir_path}/mpii_cooking/cooking.csv")
        elif self.dataset == 'epic_kitchen':
            self.all_videos = os.listdir(os.path.join(self.file_path, self.mode))
            self.annotation_df = pd.read_csv(f"{args.data_dir_path}/epic_kitchen/EPIC_100_validation.csv")
        self.transform = transform

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, index):
        return self.all_videos[index], self.load_sequence_and_target(self.all_videos[index])

    def load_sequence_and_target(self, each_video):
        img_sequence = []
        for i in range(len(os.listdir(os.path.join(self.file_path, each_video)))):
            frame = os.path.join(self.file_path, each_video, f"{i}.jpg")
            img_sequence.append(self.transform(Image.open(frame)))
        if len(img_sequence) < 16:
            pad_len = 16 - len(img_sequence)
            for j in range(pad_len):
                img_sequence.append(torch.zeros(3, 224, 224))
        img_sequence_tensor = torch.stack(img_sequence)
        img_sequence_tensor = img_sequence_tensor.permute(1, 0, 2, 3)
        target = None
        if self.dataset == 'mpii_cooking':
            selected_annotation = self.annotation_df.loc[self.annotation_df['video_id'] == each_video]
            target = self.label_map[selected_annotation['label'].values[0]]
        elif self.dataset == 'epic_kitchen':
            selected_annotation = self.annotation_df.loc[self.annotation_df['narration_id'] == each_video]
            target = selected_annotation['verb_class'].values[0]
        target = np.asarray(label_binarize([target], classes=[i for i in range(self.args.num_class)])).ravel()
        return img_sequence_tensor, torch.Tensor(target)