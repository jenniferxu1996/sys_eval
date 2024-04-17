VALID = ['s21-d21-cam-002', 's21-d23-cam-002', 's21-d27-cam-002', 's21-d28-cam-002', 's21-d29-cam-002', 's21-d35-cam-002',
         's21-d39-cam-002', 's21-d40-cam-002', 's21-d42-cam-002', 's21-d43-cam-002', 's21-d45-cam-002', 's21-d49-cam-002',
         's21-d50-cam-002', 's21-d52-cam-002', 's21-d53-cam-002', 's21-d55-cam-002', 's21-d63-cam-002']
TEST = ['s22-d23-cam-002', 's22-d25-cam-002', 's22-d26-cam-002', 's22-d29-cam-002', 's22-d31-cam-002',
        's22-d34-cam-002', 's22-d35-cam-002', 's22-d43-cam-002', 's22-d46-cam-002', 's22-d48-cam-002', 's22-d53-cam-002',
        's22-d55-cam-002', 's28-d23-cam-002', 's28-d25-cam-002', 's28-d27-cam-002', 's28-d39-cam-002', 's28-d46-cam-002',
        's28-d51-cam-002', 's28-d70-cam-002', 's28-d74-cam-002', 's29-d29-cam-002', 's29-d31-cam-002', 's29-d39-cam-002',
        's29-d42-cam-002', 's29-d49-cam-002', 's29-d50-cam-002', 's29-d52-cam-002', 's29-d71-cam-002', 's33-d23-cam-002',
        's33-d27-cam-002', 's33-d45-cam-002', 's33-d49-cam-002', 's33-d50-cam-002', 's33-d54-cam-002', 's34-d21-cam-002',
        's34-d28-cam-002', 's34-d34-cam-002', 's34-d40-cam-002', 's34-d41-cam-002', 's34-d63-cam-002', 's34-d69-cam-002',
        's34-d73-cam-002']


def label_mapping(args):
    label_map = {}
    label_path = f"{args.data_dir_path}/{args.dataset}/classes.txt"
    with open(label_path, 'r') as f:
        count = 0
        for line in f:
            label_map[line.strip()] = count
            count += 1
    return label_map