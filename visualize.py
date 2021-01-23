"""Evaluation Script"""
import os
import shutil

import tqdm
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from util.utils import set_seed
from config import ex


@ex.automain
def main(_run, _config, _log):
    # os.makedirs(f'{_run.observers[0].dir}/features', exist_ok=True)
    # for source_file, _ in _run.experiment_info['sources']:
    #     os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
    #                 exist_ok=True)
    #     _run.observers[0].save_file(source_file, f'source/{source_file}')
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    set_seed(_config['seed'])


    _log.info('###### Prepare data ######')
    dataset = _config['dataset']
    task = _config['task']
    model = _config['model']

    set_dframes = []
    exp_strs = ['_'.join([f'{ex.path}'] + [dataset,] 
        + [key for key, value in model.items() if value] 
        + [f'sets_{label_set}', f'{task["n_ways"]}way_{task["n_shots"]}shot_[test]']) for label_set in range(_config['n_sets'])]
    for exp_str in exp_strs:
        last_exp = os.listdir(f'./runs/{exp_str}')[-1]
        set_dframes.append(
            pd.concat([pd.read_csv(f'./runs/{exp_str}/{last_exp}/features/features_run_{run+1}.csv')
                for run in range(_config['n_runs'])
            ]))

    # Code for visualization works only for four sets right now,
    #  If we add a dataset with more sets, we should change the visualize script

    _log.info('###### Plotting visualization with Umap and TSNE ######')
    combinations = [[0], [1], [2], [3], [0,1], [0,2], [0,3], [1,2], [1,3], [2,3], [0,1,2], [1,2,3], [0,1,2,3]]

    for combination in combinations:
        curr_fts = pd.concat([set_dframes[comb] for comb in combination])
        comb_str = '-'.join(map(str,combnation))

        embedding = umap.UMAP().fit_transform(set_df.iloc[:, 1:])
        plt.figure(figsize=(12,12))
        plt.scatter(embedding[:, 0], embedding[:, 1], 
                    c=set_df.iloc[:, 0], 
                    edgecolor='none',
                    alpha=0.80,
                    cmap='Paired',
                    s=10)
        plt.axis('off')
        plt.savefig(f'{_run.observers[0].dir}/set_{comb_str}_Umap.png')

        # TSNE
        tsne = TSNE(n_components=2, random_state=10).fit_transform(set_df.iloc[:, 1:])
        plt.figure(figsize=(12,12))
        plt.scatter(tsne[:, 0], tsne[:, 1], 
                    c=set_df.iloc[:, 0], 
                    edgecolor='none', 
                    alpha=0.80,
                    cmap='Paired',
                    s=10)
        plt.axis('off')
        plt.savefig(f'{_run.observers[0].dir}/set_{comb_str}_TSNE.png')

        



