import matplotlib
matplotlib.use('TkAgg')
import os
import random
import numpy as np
import torch
from codes.utils import load_model
from codes.options import Options
from codes.retrieval import val_retrieval_fid, val_retrieval_csafe, \
    prepare_datasets_fid, prepare_datasets_csafe, Metric
from codes.model import SupConResNet

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main():
    opt = Options().parse()

    # Seed everything so data is sampled in consistent way
    seed_everything(opt.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)
    opt.device = torch.device('cuda:{}'.format(opt.gpu)) if opt.gpu is not None else torch.device('cpu')

    opt.in_channel = 2
    net = SupConResNet(name='resnet50', in_channel=opt.in_channel, feature_dim=opt.feature_dim).to(opt.device)
    print("Model has ", str(sum(p.numel() for p in net.parameters() if p.requires_grad) / 1e6), "M parameters")
    net, o, _, epoch = load_model(opt.weights_init, net, None)

    os.makedirs(opt.output, exist_ok=True)

    # load datasets
    val_ref_db_dataloader, val_FID_query_dataloader = prepare_datasets_fid(opt, ref_db_batch=128)
    CSAFE_val_query_dataloader = prepare_datasets_csafe(opt)
    FID_best_metric = Metric()
    CSAFE_best_metric = {category: Metric() for category in CSAFE_val_query_dataloader.dataset.category_names}

    save_images = opt.save_all_matches
    val_features = val_retrieval_fid(val_FID_query_dataloader, val_ref_db_dataloader, net, opt, save_images=save_images,
                                     best_metric=FID_best_metric)
    val_retrieval_csafe(CSAFE_val_query_dataloader, val_ref_db_dataloader, net, opt, save_images=save_images,
                        val_features=val_features, best_metric=CSAFE_best_metric)
    return

if __name__ == '__main__':
    main()
