from baselines import get_baseline
from datasets import get_dataset, list_datasets
from models import get_model, list_models

from utils.misc import *
from utils.metrics import cal_metric

from torch.utils.data import DataLoader
import argparse
import torch


def get_eval_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("--ind_dataset", type=str, default="imagenet1k")
    parser.add_argument("--ood_dataset", type=str, default="iNaturalist") 
    parser.add_argument("--model", type=str, default="resnet")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--bs", type=int, default=256)

    parser.add_argument("--ood_method", type=str, default="energy")

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    return args


if __name__ == "__main__":
    
    args = get_eval_options()
    fix_random_seed(args.random_seed)

    # get ID & OOD datasets
    ind_dataset = get_dataset(args.ind_dataset)
    ood_dataset = get_dataset(args.ood_dataset)

    # get ID & OOD loaders
    ind_loader = DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)
    ood_loader = DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)
    
    # get model
    model = get_model(args.model, num_classes=ind_dataset.num_classes, device=args.device)

    # get ood detector
    ood_detector = get_baseline(args.ood_method, model=model, args=args)

    # get id&ood scores
    ind_scores = ood_detector.eval(ind_loader)
    ood_scores = ood_detector.eval(ood_loader)

    # get metrics
    import numpy as np
    ind_labels = np.ones(ind_scores.shape[0])
    ood_labels = np.zeros(ood_scores.shape[0])

    labels = np.concatenate([ind_labels, ood_labels])
    scores = np.concatenate([ind_scores, ood_scores])

    auroc, aupr, fpr = cal_metric(labels, scores)

    print("{:10} {}".format("AUROC:", auroc))
    print("{:10} {}".format("FPR:", fpr))