
import os
import logging
import warnings
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product

import yaml

from tasks import run_downstreams, SAVE_DIR as DOWNSTREAM_DIR
from evaluation.evaluate import evaluate_score
from evaluation.evaluator import Evaluator
from utils import df_to_dict

warnings.filterwarnings('ignore')


def get_args_parser():
    # parse args
    parser = argparse.ArgumentParser(description='Set arguments for SSL-Uncertainty', add_help=False)

    # Model
    parser.add_argument('--ssl', default='simclr', choices=['simclr', 'byol', 'moco', 'byol200'])
    parser.add_argument('--arch', default='resnet18', choices=['resnet18', 'resnet50'])

    # Dataset
    parser.add_argument('--pretrain', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet32'])
    parser.add_argument('--downstream', default='cifar10', choices=['cifar10', 'cifar100', 'stl10'])

    # Paths
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--config_path', default=None, type=str, help='Path to the config file')
    # Downstream Inference
    parser.add_argument('--seed', default=0, type=int, help='Random seed deciding the reference set')
    parser.add_argument('--task_type', type=str, default='binary', choices=['binary', 'multi'])
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    return parser


def main(args):
    # Logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(levelname)s - %(message)s")

    # Load a config
    if args.config_path is None:
        config_path = f"./configs/{args.ssl}_{args.arch}_{args.pretrain}_{args.downstream}.yaml"
    else:
        config_path = args.config_path

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Run downstream tasks
    downstream_cfg = config['downstream']
    args.downstream_train_paths = [
        os.path.join(downstream_cfg['dir'], downstream_cfg['train'].format(i=i))
        for i in range(downstream_cfg['n_models'])
    ]
    args.downstream_test_paths = [
        os.path.join(downstream_cfg['dir'], downstream_cfg['test'].format(i=i))
        for i in range(downstream_cfg['n_models'])
    ]
    results_df = run_downstreams(args, load=True)

    if args.skip_evaluation:
        return

    # Evaluate the measures
    evaluator = Evaluator()
    eval_cfg = config['evaluation']
    ens_cfg = eval_cfg['deep_ens']
    for dist in ['cosine', 'euclidean']:
        # paths for loading representations
        ref_rep_paths = [
            os.path.join(ens_cfg['dir'], ens_cfg['ref'].format(j=j))
            for j in range(ens_cfg['n_ens'])
        ]
        eval_rep_paths = [
            os.path.join(ens_cfg['dir'], ens_cfg['eval'].format(j=j))
            for j in range(ens_cfg['n_ens'])
        ]

        # eval
        logging.info(f'Start evaluation with {dist} distance')
        results = evaluate_score(
            methods=eval_cfg['methods'],
            ref_rep_paths=ref_rep_paths,
            eval_rep_paths=eval_rep_paths,
            dist=dist,
            seed=args.seed,
            use_numba=True,  # choose False if numba doesn't work
            **eval_cfg['params'],
        )
        results.update_configs({
            'dist': dist,
            'seed': args.seed,
            'timestamp': datetime.now().timestamp()
        })
        evaluator.merge(results)

    # Save / load the model uuid info
    output_path = os.path.join(args.output_dir, f"{args.ssl}_{args.arch}_{args.pretrain}_{args.downstream}_{args.task_type}_seed{args.seed}")    
    if args.debug:
        df_model = evaluator.save_score_model(output_path)
    else:
        df_model = evaluator.load_model()
    
    # Evaluate the correlation
    model_id = -1  # downstream performance of ensemble
    downstream_errs = df_to_dict(results_df[results_df.model_idx == model_id].groupby('data_idx').mean()
                                 .drop(columns=['task_idx', 'model_idx']).sort_index())
    evaluator.eval(downstream_errs, model_id=model_id)
    evaluator.save_corr(df_model, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('UQ-SSL script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(DOWNSTREAM_DIR).mkdir(parents=True, exist_ok=True)
    main(args)
