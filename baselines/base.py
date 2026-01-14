from torch.utils.data import DataLoader


class BaseBaseline:

    def __init__(self, model, ind_dataset, ood_dataset, device, args):
        self.model = model
        self.ind_dataset = ind_dataset
        self.ood_dataset = ood_dataset
        self.device = device
        
        # get ID & OOD loaders
        self.ind_loader = DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)
        self.ood_loader = DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)

    def eval(self, data_loader):
        pass
