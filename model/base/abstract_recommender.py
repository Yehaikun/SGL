__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@gmail.com"

__all__ = ["AbstractRecommender"]

from reckit import Logger
from reckit import Configurator
from reckit import Evaluator
from reckit import typeassert
from data import Dataset
import abc
import time
import os
import torch
from datetime import datetime


@typeassert(config=Configurator, data_name=str)
def _create_logger(config, data_name):
    # create a logger
    log_dir = os.path.join(config.root_dir, "logs")
    log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
    logger_name = os.path.join(log_dir, log_name)
    logger = Logger(logger_name)

    return logger


class AbstractRecommender(object):
    @typeassert(config=Configurator)
    def __init__(self, config):
        self.dataset = Dataset(config.data_dir, config.dataset, config.sep, config.file_column)
        self.logger = self._create_logger(config, self.dataset)

        user_train_dict = self.dataset.train_data.to_user_dict()
        user_test_dict = self.dataset.test_data.to_user_dict()
        self.evaluator = Evaluator(self.dataset, user_train_dict, user_test_dict,
                                   metric=config.metric, top_k=config.top_k,
                                   batch_size=config.test_batch_size,
                                   num_thread=config.test_thread)

    @typeassert(config=Configurator, dataset=Dataset)
    def _create_logger(self, config, dataset):
        log_dir = os.path.join(config.root_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        try:
            gpu_id = config["gpu_id"]
        except:
            gpu_id = "?"
        log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{dataset.data_name}_gpu{gpu_id}.log"
        logger_name = os.path.join(log_dir, log_name)
        logger = Logger(logger_name)

        def cfg(name, default="N/A"):
            try:
                return config[name]
            except:
                return default

        logger.info("=" * 60)
        logger.info("                    SGL Training Log")
        logger.info("=" * 60)
        logger.info(f"PID:              {os.getpid()}")
        logger.info(f"Model:            {self.__class__.__module__}")
        logger.info(f"Model Class:      {self.__class__.__name__}")
        logger.info(f"Log Time:         {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("─" * 60)

        logger.info("[Dataset]")
        logger.info(f"  Name:            {dataset.data_name}")
        logger.info(f"  Users:           {dataset.num_users:,}")
        logger.info(f"  Items:           {dataset.num_items:,}")
        logger.info(f"  Ratings (total): {dataset.num_ratings:,}")
        logger.info(f"  Train:           {dataset.num_train_ratings:,}")
        logger.info(f"  Test:            {len(dataset.test_data):,}")
        sparsity = 100 - dataset.num_ratings / (dataset.num_users * dataset.num_items) * 100 if dataset.num_users * dataset.num_items > 0 else 0
        logger.info(f"  Sparsity:        {sparsity:.4f}%")
        logger.info("─" * 60)

        logger.info("[Model Architecture]")
        n_layers = int(cfg("n_layers"))
        embed_size = int(cfg("embed_size"))
        logger.info(f"  Base Model:       LightGCN (SGL)")
        logger.info(f"  Embedding Size:   {embed_size}")
        logger.info(f"  GCN Layers:       {n_layers}")
        logger.info(f"  Augmentation:     {cfg('aug_type')}")
        logger.info(f"  Adjacency Type:   {cfg('adj_type')}")
        logger.info(f"  Parameter Init:   {cfg('param_init')}")
        total_params = (dataset.num_users + dataset.num_items) * embed_size
        logger.info(f"  Total Params:     {total_params:,}")
        logger.info("─" * 60)

        logger.info("[Hyperparameters]")
        logger.info(f"  Batch Size:       {cfg('batch_size')}")
        logger.info(f"  Learning Rate:    {cfg('lr')}")
        logger.info(f"  Optimizer:        {cfg('learner')}")
        logger.info(f"  Reg (L2):         {cfg('reg')}")
        logger.info(f"  SSL Reg:          {cfg('ssl_reg')}")
        logger.info(f"  SSL Ratio:        {cfg('ssl_ratio')}")
        logger.info(f"  SSL Temp:         {cfg('ssl_temp')}")
        logger.info(f"  SSL Mode:         {cfg('ssl_mode')}")
        logger.info(f"  Max Epochs:       {cfg('epochs')}")
        logger.info(f"  Negative Samples: {cfg('num_negatives')}")
        logger.info(f"  Stop Patience:    {cfg('stop_cnt')}")
        logger.info("─" * 60)

        logger.info("[Hardware]")
        logger.info(f"  GPU ID:           {gpu_id}")
        logger.info(f"  CUDA Available:   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  GPU Name:         {torch.cuda.get_device_name(0)}")
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"  GPU Memory:       {total_mem:.1f} GB")
        logger.info(f"  Test Threads:     {cfg('test_thread')}")
        logger.info(f"  Test Batch Size:  {cfg('test_batch_size')}")
        logger.info("=" * 60)
        logger.info(f"Training started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)

        return logger

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def predict(self, users):
        pass
