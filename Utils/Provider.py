from Utils.paths import config_path
from Utils.ConfigParser import ConfigParser
from Utils.logging.Logging import Logging
import os


class ConfigProvider(object):
    __the_config = None

    @staticmethod
    def config():
        if ConfigProvider.__the_config is None:
            assert os.path.isfile(config_path)
            ConfigProvider.__the_config = ConfigParser(config_path).parse()
        return ConfigProvider.__the_config


