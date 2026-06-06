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
        log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
        logger_name = os.path.join(log_dir, log_name)
        logger = Logger(logger_name)

        logger.info(f"my pid: {os.getpid()}")
        logger.info(f"model: {self.__class__.__module__}")
        logger.info(self.dataset)
        logger.info(config)

        return logger

    @abc.abstractmethod
    def train_model(self):
        pass

    @abc.abstractmethod
    def predict(self, users):
        pass
