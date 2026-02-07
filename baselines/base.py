import torch
import numpy as np
import os
import mgzip
import pickle
from tqdm import tqdm

from torch.utils.data import DataLoader
from utils.tqdmfilewapper import TqdmFileWrapper

class BaseBaseline:

    def __init__(self, model, ind_dataset, ood_dataset, device, args):
        self.model = model
        self.ind_dataset = ind_dataset
        self.ood_dataset = ood_dataset
        self.device = device
        self.args = args
        
        # get ID & OOD loaders
        self.ind_loader = DataLoader(dataset=ind_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)
        self.ood_loader = DataLoader(dataset=ood_dataset, batch_size=args.bs, pin_memory=True, num_workers=8, shuffle=False)

    @torch.no_grad()
    def get_train_feature(self):
        save_path = os.path.join(self.args.cache_dir, self.args.model, self.args.ind_dataset)
        os.makedirs(save_path, exist_ok=True)
        features_file_path = os.path.join(save_path, "features.pkl")

        # =============================
        # Load feature cache if exists
        # =============================
        if os.path.exists(features_file_path):
            total_bytes = os.path.getsize(features_file_path)
            with mgzip.open(features_file_path, "rb", thread=16) as raw_f:
                wrapped_f = TqdmFileWrapper(raw_f, total_bytes, desc="Loading features")
                try:
                    save_features = pickle.load(wrapped_f)
                finally:
                    wrapped_f.close()
            return save_features 

        # =========================
        # Extract & cache features
        # =========================
        self.model.eval()
        num_classes = self.ind_dataset.num_classes
        train_features = [[] for _ in range(num_classes)]

        train_loader = DataLoader(dataset=self.ind_dataset.train_dataset, 
                                  batch_size=self.args.bs, 
                                  pin_memory=True, 
                                  num_workers=8, 
                                  shuffle=False,
                                  persistent_workers=True)
        
        for images, _ in tqdm(train_loader, desc="Extracting training features"):
            images = images.to(self.device, non_blocking=True)

            outputs = self.model.get_output(images)
            features = self.model.get_feature(images)
            p_labels = outputs.argmax(dim=1)

            feat_cpu = features.detach().to("cpu")
            p_cpu = p_labels.detach().to("cpu")

            for c in p_cpu.unique().tolist():
                mask = (p_cpu == c)
                train_features[c].append(feat_cpu[mask])
        
        save_features = []
        for c in range(num_classes):
            if len(train_features[c]) == 0:
                save_features.append(torch.empty(0))
            else:
                save_features.append(torch.cat(train_features[c], dim=0))
        
        with mgzip.open(features_file_path, "wb", thread=16) as f:
            pickle.dump(save_features, f, protocol=pickle.HIGHEST_PROTOCOL)

        return save_features
    
    def eval(self, data_loader):
        pass
