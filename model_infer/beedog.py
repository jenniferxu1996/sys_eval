import torch
import os
import numpy as np


def inference(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.context:
        node_num = args.node_num + args.obj_length
    else:
        node_num = args.node_num
    batch_size = args.batch_size

    class_path = f"data/mpii_cooking/classes.txt"
    if args.context:
        if args.f_mode == 'ftext':
            fapp_path = f"data/mpii_cooking/frame_{args.sample_num}_context_ftext_{args.node_num}_{args.model_name}_{args.n_fapp}_objlen_{args.obj_length}"
        elif args.f_mode == 'fapp':
            fapp_path = f"data/mpii_cooking/frame_{args.sample_num}_context_fapp_{args.node_num}_{args.model_name}_{args.n_fapp}_objlen_{args.obj_length}"
        else:
            fapp_path = f"data/mpii_cooking/frame_{args.sample_num}_context_fcom{args.w0}_{args.node_num}_{args.model_name}_{args.n_fapp}_objlen_{args.obj_length}"

    else:
        fapp_path = f"data/mpii_cooking/frame_{args.sample_num}_fapp_{args.node_num}_{args.model_name}_{args.n_fapp}"
    if args.fpos:
        if args.context:
            fpos_path = f"data/mpii_cooking/frame_{args.sample_num}_context_fpos_{args.node_num}_{args.n_fpos}_objlen_{args.obj_length}"
        else:
            fpos_path = f"data/mpii_cooking/frame_{args.sample_num}_fpos_{args.node_num}_{args.n_fpos}"
    else:
        fpos_path = None

    if args.graph_type == 'unigram':
        adjacency_list = generate_adj_list_unigram(node_num)
    else:
        adjacency_list = generate_adj_list_bigram(node_num)

    if args.context:
        dataset = ContextDataset(args, fapp_path, fpos_path, mode=args.infer_mode)
    else:
        dataset = Dataset(args, fapp_path, fpos_path, mode=args.infer_mode)

    dataloader = DataLoader(dataset, batch_size, num_workers=4, shuffle=False)

    classes = []
    with open(class_path, 'r') as f:
        for cls in f:
            classes.append(cls.strip())

    model = torch.load(f"result/{args.model_name}/{args.ckpt_path}").to(device)
    print(f"from check point {args.ckpt_path}")
    model.eval()

    y_ground_truth = []
    y_predict_all = []
    y_prob_all = []
    weight0_all = []
    weight1_all = []
    clip_id_all = []

    for i, (inputs, targets) in tqdm.tqdm(enumerate(dataloader), desc="inference"):
        batch_len = len(inputs)
        inputs = inputs.permute(1, 0, 2, 3)
        inputs = inputs.to(device)
        y_prob, edge_index, weight0_list, weight1_list = model(args, inputs, adjacency_list)
        targets = targets.cpu().detach().numpy()
        y_predict = np.argmax(y_prob.cpu().detach().numpy(), axis=-1)
        y_ground_truth.append(targets)
        y_prob_all.append(y_prob.cpu().detach().numpy())
        y_predict_all.append(y_predict)
        for j in range(batch_len):
            clip_id_all.append(dataset.all_samples[i * batch_size + j])
            weight0_all.append(weight0_list)
            weight1_all.append(weight1_list)

    y_ground_truth = np.concatenate(y_ground_truth)
    y_prob_all = np.concatenate(y_prob_all)
    y_predict_all = np.concatenate(y_predict_all)
    precision, recall, ap, result_report, multi_confusion, confusion, mcc, acc, f1 = get_evaluation_metrics(y_ground_truth, y_prob_all, y_predict_all, len(classes), classes)

    weight_results = pd.DataFrame(columns=['clip_id', 'weight_0', 'weight_1', 'y_predict', 'y_true'])
    for i in range(len(y_predict_all)):
        weight0 = weight0_all[i]
        weight1 = weight1_all[i]
        y_predict = y_predict_all[i]
        y_true = [j for j in range(len(y_ground_truth[i])) if y_ground_truth[i][j] == 1]
        y_true = y_true[0]
        clip_id = clip_id_all[i]
        weight_results = weight_results.append({'clip_id': clip_id, 'weight_0': weight0, 'weight_1': weight1,
                                                'y_predict': y_predict, 'y_true': y_true}, ignore_index=True)
    os.makedirs(args.output_dir, exist_ok=True)
    # weight_results.to_csv(f'{args.output_dir}/weight_results_{args.infer_mode}.csv', index=False)
    os.makedirs(f'infer/{args.output_dir}', exist_ok=True)
    weight_results.to_pickle(f'infer/{args.output_dir}/weight_results_{args.infer_mode}.pkl')

    with open(f'infer/{args.output_dir}/result_{args.infer_mode}.txt', 'w') as f:
        print("average precision:", ap, file=f)
        print("result report:", result_report, file=f)
        print("confusion matrix for multiclassification:", multi_confusion, file=f)
        print("confusion matrix:", confusion, file=f)
        print("matthews coefficient:", mcc, file=f)
        print("acc:", acc, file=f)
        print("macro f1:", f1, file=f)
