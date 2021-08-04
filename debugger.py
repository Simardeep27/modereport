import torch
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *

class DebuggerBase:
    def __init__(self,args):
        self.args=args
        self.min_val_loss = 10000000000
        self.min_val_tag_loss = 1000000
        self.min_val_stop_loss = 1000000
        self.min_val_word_loss = 10000000

        self.min_train_loss = 10000000000
        self.min_train_tag_loss = 1000000
        self.min_train_stop_loss = 1000000
        self.min_train_word_loss = 10000000

        self.params = None

        self._init_model_path()
        self.model_dir = self._init_model_dir()

        self.train_transform = self._init_train_transform()
        self.val_transform = self._init_val_transform()
        self.vocab = self._init_vocab()
        self.model_state_dict = self._load_mode_state_dict()

        self.train_data_loader = self._init_data_loader(self.args.train_file_list, self.train_transform)
        self.val_data_loader = self._init_data_loader(self.args.val_file_list, self.val_transform)

        self.extractor = self._init_visual_extractor()
        self.mlc = self._init_mlc()

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()
        self.logger = self._init_logger()

