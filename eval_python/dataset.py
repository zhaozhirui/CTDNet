import os
from easydict import EasyDict as edict


class Config_data:
    """config dataset path and value"""
    def __init__(self, cfg, dataset):
        self.current_dir = os.path.abspath('.')

        self.dataset = dataset
        self.cfg_data = edict()

        if self.dataset == 'DUT-OMRON':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'DUT-OMRON')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = None
            self.cfg_data.dataset = dataset_info
        elif self.dataset == 'DUTS':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'DUTS')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = None
            self.cfg_data.dataset = dataset_info
        elif self.dataset == 'SOD':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'SOD')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = None
            self.cfg_data.dataset = dataset_info
        elif self.dataset == 'HKU-IS':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'HKU-IS')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = None
            self.cfg_data.dataset = dataset_info
        elif self.dataset == 'ECSSD':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'ECSSD')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = None
            self.cfg_data.dataset = dataset_info
        elif self.dataset == 'PASCAL-S':
            dataset_info = edict()
            dataset_info.data_dir = os.path.join(cfg.DATA_PATH, 'PASCAL-S')
            dataset_info.gt_folder = 'mask'
            dataset_info.gt_filetype = '.png'
            self.cfg_data.dataset = dataset_info

        self.cfg_data.gt_pathlist = self._get_path_list()

    def _get_path_list(self):
        dataset = self.cfg_data.dataset

        gt_namelist = os.listdir(os.path.join(dataset.data_dir, dataset.gt_folder))
        if dataset.gt_filetype is None:
            gt_pathlist = [os.path.join(dataset.data_dir, dataset.gt_folder, filename) for filename in gt_namelist]
        else:
            gt_pathlist = [os.path.join(dataset.data_dir, dataset.gt_folder, filename) for filename in gt_namelist if filename.endswith(dataset.gt_filetype)]
        gt_pathlist.sort()

        return gt_pathlist
