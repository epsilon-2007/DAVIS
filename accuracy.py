import os
import torch
import argparse
import numpy as np
import torch.nn as nn
from configs import ModelConfig
from utils.score import ash_s, get_features, get_logits
from utils.datasets import load_in_dataset, dataset_loader, load_labeled_feature_dataset
from utils.utils import set_model, get_device, validate, accuracy, redirect_stdout_to_file, get_in_feature_path

def args_parser():
    parser = argparse.ArgumentParser(description='training module')

    parser.add_argument('--p', default=None, type=int, help='DICE prunning level')
    parser.add_argument('--ash_p', default=None, type=int, help='ASH pruning level')
    parser.add_argument('--threshold', default=None, type=float, help='ReAct threshold')
    parser.add_argument('--std', default=0.0, type=float, help='how many stndard deviation ')
    parser.add_argument('--ood_eval_type', default='avg', type=str, help=['avg', 'avg+std', 'max'])
    parser.add_argument('--ood_eval_method', default='baseline', type=str, help='[baseline, ReAct, DICE, ASH]')
    
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--checkpoint', default = 'model', type=str, help='checkpoint name')
    parser.add_argument('--num_classes', default=10, type=int, help='number of classes in in-dataset')
    parser.add_argument('--in-dataset', default="CIFAR-10", type=str, help='in-distribution dataset')
    parser.add_argument('--id_loc', default="datasets/in/", type=str, help='location of in-distribution dataset')
    parser.add_argument('--model', default='resnet18', type=str, help='model architecture: [resnet18, resnet34, densenet101, mobilenetv2]')
   
    device = get_device()
    parser.add_argument('--device', type=torch.device, default=device, help = 'device type for accelerated computation')
    args = parser.parse_args()

    args.ckpt = f"model.pt"

    if args.in_dataset in ["CIFAR-10"]:
        args.num_classes = 10
    elif args.in_dataset in ["CIFAR-100"]:
        args.num_classes = 100
    elif args.in_dataset in ["ImageNet-1K"]:
        args.num_classes = 1000

    return args

def get_accuracy(args, model, data_loader):
    model.eval()
    with torch.inference_mode():
      total = 0
      correct = 0
      for x, y in data_loader:
        x = x.to(args.device)
        y = y.to(args.device)
        x = get_features(x, model.dim_in, args)
        logits = get_logits(x, model, args)
        prob = torch.softmax(logits, dim = 1)
        pred = prob.max(dim = 1, keepdim = True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        total += len(y)
    acc = correct/total
    return acc


def main(args):

    # base directory
    base_directory_name = f"{args.model}/{args.in_dataset}/"
    model_checkpoint_directory = os.path.join(ModelConfig.model_checkpoint_directory, base_directory_name)
    model_statistics_directory = os.path.join(ModelConfig.model_statistics_directory, base_directory_name)
    model_accuracy_directory = os.path.join(ModelConfig.model_accuracy_directory, base_directory_name + f"{args.ood_eval_method}/{args.ood_eval_type}/")
    for curr_dir in [model_accuracy_directory]:
        os.makedirs(curr_dir, exist_ok=True)
    accuracy_log_writer = redirect_stdout_to_file(model_accuracy_directory, f"accuracy.log")

    info = f"{args.checkpoint}_feature_stats.npy"
    args.info = os.path.join(model_checkpoint_directory, info)
    if args.p is not None:
        args.info = np.load(args.info)

    print(args)
    print(f"acc directory: {model_accuracy_directory}")
    print(f"checkpoint directory: {model_checkpoint_directory}")

    model = set_model(args)
    # train_set, test_set = load_in_dataset(args)
    in_feature_path = get_in_feature_path(args, model_statistics_directory)
    test_set = load_labeled_feature_dataset(args, in_feature_path)

    checkpoint_loc = os.path.join(model_checkpoint_directory, args.ckpt)
    if os.path.exists(checkpoint_loc):
        print(f'loading existing model:{checkpoint_loc}')
        model.load_state_dict(torch.load(checkpoint_loc))
    

    test_loader = dataset_loader(args, test_set, batch_size=args.batch_size)
    test_accuracy = get_accuracy(args, model, test_loader)
    print("test accuracy = {:.2f}".format(test_accuracy*100))

    # loss_fn = nn.CrossEntropyLoss()
    # # evaluate model 
    # loss = 0.0
    # loss = validate(args, model, test_loader, loss_fn)
    # test_accuracy = accuracy(args, model, test_loader)
    # print("validation loss = {:.4f}, test accuracy = {:.2f}".format(loss, test_accuracy*100))

    accuracy_log_writer.close()
    

if __name__ == '__main__':
   args = args_parser() 
   main(args)