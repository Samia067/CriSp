import torch
from codes.utils import prepare_input, get_features
from codes.utils import get_shoeid_str
from codes.metric import Hit_At_K, MAP_At_K
from codes.utils import AverageMeter, save_tensor_grid, valid_tensor, get_compressed_mask
from codes.dataset.utils import image_to_channels, read_image
import codes.dataset as dataset
import torch as th
import time, os, cv2
import numpy as np
from torch.utils.data import DataLoader
from collections import deque, OrderedDict

def get_database_features(dataloader, net, opt):
    time1 = time.time()
    database_feat = None
    mask_for_image = None
    db_feat_device = th.device("cpu")
    for idx, data in enumerate(dataloader):
        image, mask, shoeprint, albedo, depth, name = data
        image, mask, shoeprint, albedo, depth = [item.to(opt.device) if valid_tensor(item) else item for item in
                                                 [image, mask, shoeprint, albedo, depth]]

        inp = prepare_input(True, mask_for_image, opt, image, depth, shoeprint)
        features = get_features(inp, net, spatial_feat=True)

        if database_feat is None:
            database_feat = th.zeros([len(dataloader.dataset)] + list(features.shape[1:])).to(db_feat_device)
            val_feature_index = 0

        bsz = image.shape[0]
        database_feat[val_feature_index: min(val_feature_index + bsz, database_feat.shape[0])] = features.to(db_feat_device)
        val_feature_index += bsz
        print('Dataset feature calculation: ' + str(int((idx + 1) / len(dataloader) * 100)) + ' % complete', end='\r')
    print('Dataset feature calculation time: {:.2f}\t'.format(time.time() - time1))
    return database_feat

def prepare_datasets_fid(opt, ref_db_batch=256, query_batch=1):
    database_dir = opt.database_dir
    val_ref_db_dataset = dataset.RefDb(os.path.join(opt.dataroot, database_dir), opt)
    val_ref_db_dataloader = DataLoader(val_ref_db_dataset, batch_size=ref_db_batch, shuffle=False, num_workers=opt.num_workers)
    val_FID_query_dataset = dataset.FID300(os.path.join(opt.dataroot, opt.query_dir_FID))
    val_FID_query_dataloader = DataLoader(val_FID_query_dataset, batch_size=query_batch, shuffle=False, num_workers=opt.num_workers)

    return val_ref_db_dataloader, val_FID_query_dataloader


def prepare_datasets_csafe(opt, query_batch=1):
    csafe_query_dataset = dataset.CSAFE(os.path.join(opt.dataroot, opt.query_dir_ShoeCase))
    csafe_query_dataloader = DataLoader(csafe_query_dataset, batch_size=query_batch, shuffle=False, num_workers=opt.num_workers)
    return csafe_query_dataloader


def query_data_unload(data, opt):
    shoeprint, mask, name = data
    shoeprint = shoeprint.to(opt.device)
    mask = mask.to(opt.device)
    image = depth = None
    inp = prepare_input(False, None, opt, image, depth, shoeprint)
    return inp, name, mask

def get_sorted_matches(val_features, features):
    query_dot_dataset = th.matmul(features, val_features.T)
    values, indices = th.sort(query_dot_dataset, dim=1, descending=True)
    return values, indices

def get_score(values, indices, name, inp, all_query_matches, map_at_k, hit_at_k, opt,
              real_val_dataset, output_shape=[8,5], save_images=False, val_set='FID'):
    max_matches = (output_shape[0] * output_shape[1]) - 1
    results = {}
    for shoe_index in range(len(indices)):
        if save_images:
            visuals = OrderedDict()

            if torch.sum(inp[shoe_index:shoe_index + 1, 0:1, ...]) > 0.00000001:
                visuals['query: ' + name[shoe_index]] = inp[shoe_index:shoe_index + 1, 0:1, ...]
            else:
                visuals['query: ' + name[shoe_index]] = inp[shoe_index:shoe_index + 1, 1:, ...]

        matches = set()
        ranked_matches = deque()
        shoe_results = deque()
        match_index = 0

        if save_images:
            base_write_dir = os.path.join(opt.output, val_set)
            detail_write_dir = os.path.join(base_write_dir, name[shoe_index].split('.')[0])
            os.makedirs(base_write_dir, exist_ok=True)
            os.makedirs(detail_write_dir, exist_ok=True)

        while len(matches) < max(100, max_matches):

            cur_ind = indices[shoe_index, match_index]
            cur_val = values[shoe_index, match_index]

            key = real_val_dataset.image_file_names[cur_ind]
            shoeid = get_shoeid_str(key)
            match_index += 1
            if shoeid in matches:
                continue

            matches.add(shoeid)

            if save_images:
                img = th.tensor(image_to_channels(read_image(real_val_dataset.image_files[cur_ind]))).unsqueeze(0)
                impression = th.tensor(image_to_channels(read_image(real_val_dataset.print_files[cur_ind]))).unsqueeze(0)
                img = th.cat((img, impression), dim=3)
                key = str(len(matches)).zfill(4) + "_" + '%.4f' % cur_val.item() + '_' + key

                if len(matches) <= max_matches:
                    visuals[key] = img

                image_to_save = img.cpu().detach().numpy()[0, ...].transpose(1, 2, 0) * 255
                cv2.imwrite(os.path.join(detail_write_dir, key), image_to_save)

            ranked_matches.append(shoeid)
            shoe_results.append(key)
        results[name[shoe_index]] = shoe_results

        if all_query_matches:
            if name[shoe_index][-4] == '.':
                true_matches = all_query_matches[name[shoe_index][:-4]]
            else:
                true_matches = all_query_matches[name[shoe_index]]
            if len(true_matches) > 0:
                map_at_k.add(ranked_matches, true_matches)
                hit_at_k.add(ranked_matches, true_matches)

        if save_images:
            save_tensor_grid(visuals, detail_write_dir, figsize=(40, 20), fig_shape=output_shape)

    return results

def run_query(val_query_dataloader, val_ref_db_dataset, database_feat, net, opt, query_matches, max_rank=100,
              save_images=False, val_set='FID'):
    time1 = time.time()
    map_at_k = MAP_At_K(max_rank)
    hit_at_k = Hit_At_K(max_rank)

    results = {}
    for query_idx, data in enumerate(val_query_dataloader):
        query_inp, name, mask = query_data_unload(data, opt)

        query_feat = get_features(query_inp, net, spatial_feat=True)
        compressed_mask = get_compressed_mask(mask)
        query_feat = net.vectorize((query_feat*compressed_mask.to(query_feat.device)).to(query_inp.device), spatial=True)

        values, indices = get_sorted_matches(database_feat, query_feat)
        output_shape = [8, 5]
        res = get_score(values, indices, name, query_inp, query_matches, map_at_k, hit_at_k, opt,
                        val_ref_db_dataset, output_shape=output_shape,
                        save_images=save_images, val_set=val_set)
        results.update(res)

    print('Query time for {} items: {:.2f}'.format(len(val_query_dataloader.dataset), time.time() - time1))
    return map_at_k, hit_at_k, results


def val_retrieval(val_ref_db_dataloader, val_query_dataloader, net, opt, query_matches,
                  database_feat=None, save_images=False, val_set='FID'):
    net.eval()
    with th.no_grad():
        dataset_time = AverageMeter()
        start_time = time.time()

        if database_feat is None:
            if (opt.saved_val_features is not None) and os.path.exists(opt.saved_val_features):
                database_feat = th.load(opt.saved_val_features).to(opt.device)
            else:
                database_feat = get_database_features(val_ref_db_dataloader, net, opt)
                database_feat = net.vectorize(database_feat, spatial=True).to(opt.device)
                th.save(database_feat, opt.saved_val_features)

        dataset_time.update(time.time() - start_time)
        start_time = time.time()
        query_time = AverageMeter()
        max_rank = 100 # k for map@k

        map_at_k, hit_at_k, results = \
            run_query(val_query_dataloader, val_ref_db_dataloader.dataset, database_feat, net, opt, query_matches,
                      max_rank=max_rank, save_images=save_images, val_set=val_set)

        query_time.update(time.time() - start_time)
        average_hit, hit_at_ks = hit_at_k.get_scores()
        average_map, map_at_ks = map_at_k.get_scores()

    return database_feat, Metric(hit_at_ks[99], map_at_ks[99])

def val_retrieval_csafe(CSAFE_val_query_dataloader, val_dataloader, net, opt, save_images=False,
                        val_features=None, best_metric=None):
    CSAFE_matches = CSAFE_val_query_dataloader.dataset.get_all_matches()
    similar_metrics = [AverageMeter() for i in range(len(Metric.names))]

    for category in CSAFE_val_query_dataloader.dataset.get_categories():
        CSAFE_val_query_dataloader.dataset.set_category(category)
        val_features, metrics = val_retrieval(val_dataloader, CSAFE_val_query_dataloader, net,
                                                          opt, CSAFE_matches, val_set='CSAFE',
                                                          save_images=save_images, database_feat=val_features)
        print(f"ShoeCase {category} metrics: {metrics.to_str()}")
        if best_metric is not None:
            best_metric[category].update_best(metrics)

        for i, m in enumerate(metrics.get_all()):
            similar_metrics[i].update(m)

    return val_features


def val_retrieval_fid(val_FID_crime_query_dataloader, val_ref_db_dataloader, net, opt, save_images=False,
                      database_feat=None, best_metric=None):
    FID_crime_matches = val_FID_crime_query_dataloader.dataset.get_all_matches()
    database_feat, metrics = val_retrieval(val_ref_db_dataloader, val_FID_crime_query_dataloader, net, opt,
                                           FID_crime_matches, database_feat=database_feat,
                                           save_images=save_images, val_set='FID')
    print(f"FID metrics: {metrics.to_str()}")
    if best_metric is not None:
        best_metric.update_best(metrics)

    return database_feat


class Metric:
    names = ['hit@100', 'map@100']
    def __init__(self, hit=0, map=0):
        self.hit = hit
        self.map = map

    def get_all(self):
        return (self.hit, self.map)

    def to_str(self):
        return 'hit@100 {:.2f}, \tmap@100 {:.2f}'.format(self.hit*100, self.map*100)

    def update_best(self, metric):
        self.hit = max(self.hit, metric.hit)
        self.map = max(self.map, metric.map)
