from baselines import get_baseline
from datasets import get_dataset, list_datasets
from models import get_model, list_models

from utils.misc import *
from utils.metrics import cal_metric

import argparse
import torch


def get_eval_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="imagenet1k")
    parser.add_argument("--ood_dataset", type=str, default="iNaturalist") 
    parser.add_argument("--model", type=str, default="clip")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=256)

    parser.add_argument("--ood_method", type=str, default="mcm")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    return args


if __name__ == "__main__":
    
    args = get_eval_options()
    fix_random_seed(args.random_seed)

    # get ID & OOD datasets
    ind_dataset = get_dataset(args.ind_dataset)
    ood_dataset = get_dataset(args.ood_dataset)
    
    # get model
    model = get_model(args.model, num_classes=ind_dataset.num_classes, device=args.device)

    # get ood detector
    ood_detector = get_baseline(args.ood_method, 
                                model=model, 
                                ind_dataset=ind_dataset, 
                                ood_dataset=ood_dataset, 
                                device=args.device,
                                args=args)

    # get id&ood scores
    ind_scores = ood_detector.eval(ood_detector.ind_loader)
    ood_scores = ood_detector.eval(ood_detector.ood_loader)

    # get metrics
    auroc, aupr, fpr = cal_metric(ind_scores, ood_scores)
    print("{:10} {}".format("AUROC:", auroc))
    print("{:10} {}".format("FPR:", fpr))