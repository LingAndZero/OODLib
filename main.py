from baselines import get_baseline
from datasets import get_dataset, list_datasets
from models import get_model, list_models

from utils.options import get_eval_options
from utils.misc import *
from torch.utils.data import DataLoader

from tqdm import tqdm


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
    model = get_model(args.model, num_classes=ind_dataset.num_classes)

    for (images, _) in tqdm(ind_loader):
        images = images.cuda()
        
        output = model.get_output(images)
