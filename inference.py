import os
import pandas as pd
import numpy as np
import tqdm
from model_infer import beedog, conv_lstm, single_frame, vivit, resnet_3d
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, multilabel_confusion_matrix, matthews_corrcoef, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import argparse


def get_average_precision_value(y_true, y_prob, class_num):
    precision = {}
    recall = {}
    ap = {}
    for i in range(class_num):
        print(i)
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_prob[:, i])
        ap[i] = average_precision_score(y_true[:, i], y_prob[:, i])
        # print(type(precision[i]), type(recall[i]), type(ap[i]))
        precision[i] = list(precision[i])
        recall[i] = list(recall[i])
        ap[i] = float(ap[i])
    return precision, recall, ap


def get_evaluation_metrics(y_true, y_prob, y_predict, class_num):
    # Get all the evaluation metrics for the inference data sample
    precision_all, recall_all, ap = get_average_precision_value(y_true, y_prob, class_num)
    # Convert the prediction sequence to one-hot representation
    y_predict_onehot = label_binarize(y_predict, classes=[i for i in range(class_num)])
    multi_confusion = multilabel_confusion_matrix(y_true, y_predict_onehot)
    # print(f"the shape of multi_confusion = {len(multi_confusion)}")
    # Convert the one-hot representation for ground truth to the label sequence
    y_true_sequence = []
    for i in range(len(y_true)):
        for label in range(0, class_num):
            if y_true[i][label] == 1:
                y_true_sequence.append(label)
    # print(y_true_sequence)
    labels = [i for i in range(0, class_num)]
    result_report = classification_report(y_true_sequence, y_predict, labels=labels)
    confusion = confusion_matrix(y_true_sequence, y_predict, labels=labels)
    mcc = matthews_corrcoef(y_true_sequence, y_predict)
    f1 = f1_score(y_true_sequence, y_predict, average="weighted")
    acc = accuracy_score(y_true_sequence, y_predict)
    # print(type(precision), type(recall), type(ap), type(multi_confusion), type(confusion))
    return precision_all, recall_all, ap, result_report, multi_confusion.tolist(), confusion.tolist(), mcc, acc, f1

def inference_main(args):
    id_all = None
    gt_all = None
    pred_all = None
    prob_all = None
    fc_all = None

    if args.model == 'beedog':
        id_all, gt_all, pred_all, prob_all, fc_all = beedog.inference(args)
    elif args.model == 'conv_lstm':
        id_all, gt_all, pred_all, prob_all, fc_all = conv_lstm.inference(args)
    elif args.model == 'single_frame':
        id_all, gt_all, pred_all, prob_all, fc_all = single_frame.inference(args)
    elif args.model == 'resnet18_3d' or args.model == 'resnet50_3d':
        id_all, gt_all, pred_all, prob_all, fc_all = resnet_3d.inference(args)
    elif args.model == 'vivit':
        id_all, gt_all, pred_all, prob_all, fc_all = vivit.inference(args)

    inference_res = pd.DataFrame({'id': id_all, 'gt': gt_all, 'pred': pred_all, 'conf': prob_all, 'logit': fc_all})
    if args.pretrained:
        inference_res.to_csv(f'inference_results/{args.model}_{args.pretrained_set}_pre_{args.mode}.csv', index=False)
        inference_res.to_pickle(f'inference_results/{args.model}_{args.pretrained_set}_pre_{args.mode}.pkl')
    else:
        inference_res.to_csv(f'inference_results/{args.model}_{args.mode}.csv', index=False)
        inference_res.to_pickle(f'inference_results/{args.model}_{args.mode}.pkl')

    labels = [i for i in range(0, args.num_class)]
    result_report = classification_report(np.ravel(np.array(gt_all)), np.ravel(np.array(pred_all)), labels=labels)
    print(result_report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='single_frame', type=str)
    parser.add_argument('--dataset', default='mpii_cooking', type=str)
    parser.add_argument('--data_dir_path', default="../data")
    parser.add_argument('--sampled_frame', default=8)
    parser.add_argument('--num_class', default=87, type=int)
    parser.add_argument('--pretrained', default=False, type=bool)
    parser.add_argument('--pretrained_set', default='k', choices=['k', 'km'])
    parser.add_argument('--mode', default='validation', choices=['train', 'validation'])
    args = parser.parse_args()

    inference_main(args)



