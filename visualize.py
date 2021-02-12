"""Evaluation Script"""
import os
import shutil
import csv
import json

import tqdm
import numpy as np
import pandas as pd

from util.utils import set_seed
from util.visual import plot_umap, plot_tsne
from config import ex


@ex.automain
def main(_run, _config, _log):
    # Code for visualization works only for four sets right now,
    #  If we add a dataset with more sets, we should change the visualize script
    os.makedirs(f'{_run.observers[0].dir}/TSNE', exist_ok=True)
    os.makedirs(f'{_run.observers[0].dir}/Umap', exist_ok=True)
    shutil.rmtree(f'{_run.observers[0].basedir}/_sources')
    set_seed(_config['seed'])  

    _log.info('###### Prepare data ######')
    dataset = _config['dataset']
    task = _config['task']
    model = _config['model']

    lbl_sets_paths = ['_'.join([f'{ex.path}'] + [dataset,] + ['[test]']
        + [key for key, value in model.items() if value] 
        + [f'{task["n_ways"]}way_{task["n_shots"]}shot_split{label_set}']) for label_set in range(_config['n_sets'])]

    metrics = []
    set_dframes = []
    split_paths = []
    for lbl_set_path in lbl_sets_paths:
        experiment_id = _config['experiment_id']
        if experiment_id == 'last':
            experiment_id = os.listdir(f'./runs/{lbl_set_path}')[-1]
        split_paths.append(f'{lbl_set_path}/{experiment_id}/')

        # Obtaining features
        set_dframes.append(
            pd.concat([pd.read_csv(f'./runs/{lbl_set_path}/{experiment_id}/features/features_run_{run+1}.csv')
                for run in range(_config['n_runs'])
            ]))

        # Obtaining metrics
        with open(f'./runs/{lbl_set_path}/{experiment_id}/metrics.json') as f:
            data = json.load(f)
            metrics.append(
                data['final_meanIoU']['values'][0]
            )

    _log.info('###### Saving metrics from all sets ######')
    experiment_name = '_'.join(
        [f'{ex.path}'] + [dataset,]
        + [key for key, value in model.items() if value]
        + [f'{task["n_ways"]}way_{task["n_shots"]}shot'])
    
    # Save in experiment folder
    columns = ['Name', 'Details', 'Split1', 'Split2', 'Split3', 'Split4', 
    'MeanIoU', 'n_ways', 'n_shots', 'n_queries', 'TestSeed', 'TrainingSeed',
    'Split1_path', 'Split2_path', 'Split3_path', 'Split4_path',]
    row = [experiment_name, _config['details']] \
        + metrics \
        + [sum(metrics) / len(metrics), task["n_ways"], task["n_shots"], 
            task["n_queries"], _config['seed'], _config['train_seed']] \
        + split_paths
        
    features_df = pd.DataFrame([row], columns=columns)  
    features_df.to_csv(f'{_run.observers[0].dir}/metrics.csv', index=False)

    if os.path.exists('metrics.csv'):
        results = pd.read_csv('metrics.csv')
        results = pd.concat([results, features_df], ignore_index=True)
        results.sort_values(by=['Name', 'MeanIoU'], inplace=True)
    else:
        results = features_df

    results.to_csv('metrics.csv', index=False)


    _log.info('###### Plotting visualization with Umap and TSNE ######')
    combinations = [[0], [1], [2], [3], [0,1], [0,2], [0,3], 
                    [1,2], [1,3], [2,3], [0,1,2], [1,2,3], [0,1,2,3]]

    for combination in combinations:
        curr_fts = pd.concat([set_dframes[comb] for comb in combination])
        curr_fts = curr_fts.drop_duplicates(subset=['id'])
        comb_str = '-'.join(map(str,combination))

        _log.info(f'### Obtaining Umap for label sets {comb_str} ###')
        plot_umap(curr_fts, f'{_run.observers[0].dir}/Umap/set_{comb_str}_Umap.png')

        # TSNE
        _log.info(f'### Obtaining TSNE for label sets {comb_str} ###')
        plot_tsne(curr_fts, f'{_run.observers[0].dir}/TSNE/set_{comb_str}_TSNE.png')

        



