"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose

from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import ToTensorNormalize
from dataloaders.transforms import Resize
from util.metric import Metric
from util.utils import set_seed, CLASS_LABELS
from config import ex


@ex.automain
def main(_run, _config, _log):
    os.makedirs(f'{_run.observers[0].dir}/features', exist_ok=True)
    for source_file, _ in _run.experiment_info['sources']:
        os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                    exist_ok=True)
        _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)


    _log.info('###### Create model ######')
    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'])
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'],])
    if not _config['notrain']:
        model.load_state_dict(torch.load(_config['snapshot'], map_location='cpu'))
    model.eval()


    _log.info('###### Prepare data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
        max_label = 20
    elif data_name == 'COCO':
        make_data = coco_fewshot
        max_label = 80
    else:
        raise ValueError('Wrong config for dataset!')

    labels = CLASS_LABELS[data_name]['all'] - CLASS_LABELS[data_name][_config['label_sets']]
    transforms = [Resize(size=_config['input_size'])]
    transforms = Compose(transforms)


    _log.info('###### Testing begins ######')
    metric = Metric(max_label=max_label, n_runs=_config['n_runs'])
    with torch.no_grad():
        for run in range(_config['n_runs']):
            _log.info(f'### Run {run + 1} ###')
            set_seed(_config['seed'] + run)
            features_df = pd.DataFrame()

            _log.info(f'### Load data ###')
            dataset = make_data(
                base_dir=_config['path'][data_name]['data_dir'],
                split=_config['path'][data_name]['data_split'],
                transforms=transforms,
                to_tensor=ToTensorNormalize(),
                labels=labels,
                max_iters=_config['n_steps'] * _config['batch_size'],
                n_ways=_config['task']['n_ways'],
                n_shots=_config['task']['n_shots'],
                n_queries=_config['task']['n_queries']
            )

            if _config['dataset'] == 'COCO':
                coco_cls_ids = dataset.datasets[0].dataset.coco.getCatIds()

            testloader = DataLoader(dataset, batch_size=_config['batch_size'], shuffle=False,
                                    num_workers=1, pin_memory=True, drop_last=False)
            _log.info(f"Total # of Data: {len(dataset)}")


            for sample_batched in tqdm.tqdm(testloader):
                if _config['dataset'] == 'COCO':
                    label_ids = [coco_cls_ids.index(x) + 1 for x in sample_batched['class_ids']]
                else:
                    label_ids = list(sample_batched['class_ids'])
                    
                support_images = [[shot.cuda() for shot in way]
                                  for way in sample_batched['support_images']]

                support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                                    for way in sample_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                                    for way in sample_batched['support_mask']]

                query_images = [query_image.cuda()
                                for query_image in sample_batched['query_images']]
                query_labels = torch.cat(
                    [query_label.cuda()for query_label in sample_batched['query_labels']], dim=0)

                query_pred, _, supp_fts = model(support_images, support_fg_mask, support_bg_mask,
                                      query_images)

                metric.record(np.array(query_pred.argmax(dim=1)[0].cpu()),
                              np.array(query_labels[0].cpu()),
                              labels=label_ids, n_run=run)
                
                df_fts = [features_df]
                for i, label_id in enumerate(label_ids):
                    lbl_df = pd.DataFrame(torch.cat(supp_fts[i]).cpu().numpy())
                    lbl_df['label'] = label_id.item()
                    df_fts.append(lbl_df)

                features_df = pd.concat(df_fts)

            classIoU, meanIoU = metric.get_mIoU(labels=sorted(labels), n_run=run)
            classIoU_binary, meanIoU_binary = metric.get_mIoU_binary(n_run=run)

            _run.log_scalar('classIoU', classIoU.tolist())
            _run.log_scalar('meanIoU', meanIoU.tolist())
            _run.log_scalar('classIoU_binary', classIoU_binary.tolist())
            _run.log_scalar('meanIoU_binary', meanIoU_binary.tolist())
            _log.info(f'classIoU: {classIoU}')
            _log.info(f'meanIoU: {meanIoU}')
            _log.info(f'classIoU_binary: {classIoU_binary}')
            _log.info(f'meanIoU_binary: {meanIoU_binary}')

            _log.info('Exporting features CSV')
            cols = list(features_df)
            cols = [cols[-1]] + cols[:-1]
            features_df = features_df[cols]
            features_df.to_csv(f'{_run.observers[0].dir}/features/features_run_{run+1}.csv', index=False)

    classIoU, classIoU_std, meanIoU, meanIoU_std = metric.get_mIoU(labels=sorted(labels))
    classIoU_binary, classIoU_std_binary, meanIoU_binary, meanIoU_std_binary = metric.get_mIoU_binary()

    _log.info('###### Saving features visualization ######')
    all_fts = pd.concat([pd.read_csv(f'{_run.observers[0].dir}/features/features_run_{run+1}.csv') for run in range(_config['n_runs'])])
    # Umap
    embedding = umap.UMAP().fit_transform(all_fts.iloc[:, 1:])
    plt.figure(figsize=(12,12))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=all_fts.iloc[:, 0], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Set1',
                s=5)
    plt.axis('off')
    plt.savefig(f'{_run.observers[0].dir}/features/Umap_fts.png')

    # TSNE
    tsne = TSNE(n_components=2, random_state=10).fit_transform(all_fts.iloc[:, 1:])
    plt.figure(figsize=(12,12))
    plt.scatter(embedding[:, 0], embedding[:, 1], 
                c=all_fts.iloc[:, 0], 
                edgecolor='none', 
                alpha=0.80,
                cmap='Set1',
                s=5)
    plt.axis('off')
    plt.savefig(f'{_run.observers[0].dir}/features/TSNE_fts.png')


    _log.info('----- Final Result -----')
    _run.log_scalar('final_classIoU', classIoU.tolist())
    _run.log_scalar('final_classIoU_std', classIoU_std.tolist())
    _run.log_scalar('final_meanIoU', meanIoU.tolist())
    _run.log_scalar('final_meanIoU_std', meanIoU_std.tolist())
    _run.log_scalar('final_classIoU_binary', classIoU_binary.tolist())
    _run.log_scalar('final_classIoU_std_binary', classIoU_std_binary.tolist())
    _run.log_scalar('final_meanIoU_binary', meanIoU_binary.tolist())
    _run.log_scalar('final_meanIoU_std_binary', meanIoU_std_binary.tolist())
    _log.info(f'classIoU mean: {classIoU}')
    _log.info(f'classIoU std: {classIoU_std}')
    _log.info(f'meanIoU mean: {meanIoU}')
    _log.info(f'meanIoU std: {meanIoU_std}')
    _log.info(f'classIoU_binary mean: {classIoU_binary}')
    _log.info(f'classIoU_binary std: {classIoU_std_binary}')
    _log.info(f'meanIoU_binary mean: {meanIoU_binary}')
    _log.info(f'meanIoU_binary std: {meanIoU_std_binary}')


