import os
import yaml
from easydict import EasyDict as edict
from argparse import ArgumentParser
import cv2
import numpy as np
import pandas as pd
from dataset import Config_data
from metrics import MAE, Fmeasure, wFmeasure, Smeasure, Emeasure, PRCurve


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('path', type=str, help='Path to the saliency map folder')
    return parser.parse_args()


def get_config_from_file():
    f = open('config.yaml', 'r')
    cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    f.close()
    cfg = edict(cfg)
    return cfg


if __name__ == '__main__':
    cfg = get_config_from_file()
    args = parse_args()
    cfg.EVAL_PATH = args.path

    if cfg.SAVE_PATH is None:
        cfg.SAVE_PATH = cfg.EVAL_PATH
    if not os.path.exists(cfg.SAVE_PATH):
        os.makedirs(cfg.SAVE_PATH)

    global_results_list = []
    if isinstance(cfg.DATASETS, str):
        cfg.DATASETS = [cfg.DATASETS]
    for dataset in cfg.DATASETS:
        dataset_result = edict()
        dataset_result.dataset = dataset
        dataset_result.mae = 0
        dataset_result.fm = 0
        dataset_result.wfm = 0
        dataset_result.sm = 0
        dataset_result.em = 0
        dataset_results_list = []
        prec = np.zeros(256)
        rec = np.zeros(256)

        pred_path = os.path.join(cfg.EVAL_PATH, dataset)
        if not os.path.exists(pred_path):
            continue
        pred_list = os.listdir(pred_path)
        pred_list = [os.path.join(pred_path, filename) for filename in pred_list]
        pred_list.sort()

        obj_cfg = Config_data(cfg, dataset)
        mask_list = obj_cfg.cfg_data.gt_pathlist
        mask_num = len(mask_list)
        assert len(pred_list) == mask_num, 'The number of predicted saliency maps and ground truths of dataset %s are not equal' % dataset

        for i in range(mask_num):
            fg = cv2.imread(pred_list[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(mask_list[i], cv2.IMREAD_GRAYSCALE)
            fg = (fg - fg.min()) / (fg.max() - fg.min() + np.finfo(np.float64).eps)
            gt = gt > 127
            if fg.max() == 0 or gt.max() == 0:
                continue

            result = edict()
            result.filename = os.path.basename(mask_list[i])
            result.mae = MAE(fg, gt)
            result.fm = Fmeasure(fg, gt)[2]
            result.wfm = wFmeasure(fg, gt)
            result.sm = Smeasure(fg, gt)
            result.em = Emeasure(fg, gt)
            dataset_results_list.append(result)

            dataset_result.mae += result.mae
            dataset_result.fm += result.fm
            dataset_result.wfm += result.wfm
            dataset_result.sm += result.sm
            dataset_result.em += result.em
            precision, recall = PRCurve(fg * 255, gt)
            prec += precision
            rec += recall

        prec /= mask_num
        rec /= mask_num
        beta = 0.3
        dataset_result.maxF = np.max((1 + beta) * prec * rec / (beta * prec + rec + np.finfo(np.float64).eps))
        dataset_result.mae /= mask_num
        dataset_result.fm /= mask_num
        dataset_result.wfm /= mask_num
        dataset_result.sm /= mask_num
        dataset_result.em /= mask_num
        global_results_list.append(dataset_result)

        if cfg.SAVE_PATH is not None:
            dataset_results_table = pd.DataFrame(dataset_results_list)
            with open(os.path.join(cfg.SAVE_PATH, '%s_dataset_result.csv' % dataset), 'w') as f:
                dataset_results_table.to_csv(f, index=False, float_format='%.3f')

    global_results_table = pd.DataFrame(global_results_list)
    print(global_results_table.to_string(index=False, float_format='%.3f'))
    if cfg.SAVE_PATH is not None:
        with open(os.path.join(cfg.SAVE_PATH, 'global_result.csv'), 'w') as f:
            global_results_table.to_csv(f, index=False, float_format='%.3f')
