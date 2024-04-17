import json
import os
from types import SimpleNamespace

import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
from sklearn.preprocessing import label_binarize
from torch.utils import data
from PIL import Image
from model.vivit.model_trainer import VideoTransformer
import model_infer.utils_mpii_cooking as utils_mpii_cooking


class ViViTDataloader(data.Dataset):
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
        return self.all_videos[index], self.load_frame_and_target(self.all_videos[index])


    def load_frame_and_target(self, each_video):
        img_sequence = []
        for i in range(len(os.listdir(os.path.join(self.file_path, each_video)))):
            frame = os.path.join(self.file_path, each_video, f"{i}.jpg")
            img_sequence.append(self.transform(Image.open(frame)))
        if len(img_sequence) < 16:
            pad_len = 16 - len(img_sequence)
            for j in range(pad_len):
                img_sequence.append(torch.zeros(3, 224, 224))
        img_sequence_tensor = torch.stack(img_sequence)
        selected_annotation = self.annotation_df.loc[self.annotation_df['video_id'] == each_video]
        target = self.label_map[selected_annotation['label'].values[0]]
        target = np.asarray(label_binarize([target], classes=[i for i in range(self.args.num_class)])).ravel()
        return img_sequence_tensor, torch.Tensor(target)


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    classes = []
    data_dir_path = args.data_dir_path
    infer_data = ViViTDataloader(args, transform=transform)
    if args.dataset == 'mpii_cooking':
        class_path = f"{data_dir_path}/mpii_cooking/classes.txt"
        with open(class_path, 'r') as f:
            for cls in f:
                classes.append(cls.strip())
    elif args.dataset == "epic_kitchen":
        cls_df = pd.read_csv(f"{data_dir_path}/epic_kitchen/EPIC_100_verb_classes.csv")
        for i, row in cls_df.iterrows():
            classes.append(row['key'])

    infer_loader = DataLoader(infer_data, 1, num_workers=4, shuffle=False)

    with open(f"ckpt_repo/{args.dataset}/vivit_config.json", 'r') as f:
        configs = json.load(f, object_hook=lambda d: SimpleNamespace(**d))

    model = VideoTransformer.load_from_checkpoint(f"ckpt_repo/{args.dataset}/vivit.pth", configs=configs).to(device)
    print(f"from check point ckpt_repo/{args.dataset}/vivit.pth")

    model.eval()

    # id_all = ResnetDataloader.get_each_video_id(infer_data)
    id_all = []
    gt_all = []
    prob_all = []
    pred_all = []
    fc_all = []

    for video, (inputs, targets) in tqdm.tqdm(infer_loader, desc="inference"):
        id_all.append(video[0])
        inputs = inputs.to(device)
        gt_all.append(np.argmax(targets, axis=-1).numpy())
        logits = model.validate(inputs)
        fc_all.append(np.ravel(logits.cpu().detach().numpy()))
        y_prob = F.softmax(model.validate(inputs), dim=1)
        # y_prob = model(inputs)
        prob_all.append(np.ravel(y_prob.cpu().detach().numpy()))
        y_predict = np.argmax(y_prob.cpu().detach().numpy(), axis=-1)
        pred_all.append(y_predict)

    return id_all, gt_all, pred_all, prob_all, fc_all

