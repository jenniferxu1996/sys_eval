import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm
import torch
from data_loader import SeqDataloader


def inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match the expected input size
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    data_dir_path = args.data_dir_path
    infer_data = SeqDataloader(args, transform=transform)

    classes = []
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

    if args.pretrained:
        model = torch.load(f"ckpt_repo/{args.dataset}/conv_lstm_pre.pt").to(device)
        print(f"from check point ckpt_repo/{args.dataset}/conv_lstm_pre.pt")
    else:
        model = torch.load(f"ckpt_repo/{args.dataset}/conv_lstm.pt").to(device)
        print(f"from check point ckpt_repo/{args.dataset}/conv_lstm.pt")

    model.eval()

    id_all = []
    gt_all = []
    prob_all = []
    pred_all = []
    fc_all = []

    for video, (inputs, targets) in tqdm.tqdm(infer_loader, desc="inference"):
        id_all.append(video[0])
        inputs = inputs.to(device)
        gt_all.append(np.argmax(targets, axis=-1).numpy())
        logits = model(inputs)
        fc_all.append(np.ravel(logits.cpu().detach().numpy()))
        y_prob = F.softmax(model(inputs), dim=1)
        # y_prob = model(inputs)
        prob_all.append(np.ravel(y_prob.cpu().detach().numpy()))
        y_predict = np.argmax(y_prob.cpu().detach().numpy(), axis=-1)
        pred_all.append(y_predict)

    return id_all, gt_all, pred_all, prob_all, fc_all