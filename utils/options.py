import argparse
import torch


def get_eval_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="imagenet1k")
    parser.add_argument("--ood_dataset", type=str, default="iNaturalist") 
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=32)

    parser.add_argument("--ood_method", type=str, default="NegLabel")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    return args