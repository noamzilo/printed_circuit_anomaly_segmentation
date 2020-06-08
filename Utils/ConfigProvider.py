from Utils.paths import config_path
from Utils.ConfigParser import ConfigParser
from Utils.logging.Logging import Logging
import os


class ConfigProvider(object):
    __the_config = None

    @staticmethod
    def config():
        if ConfigProvider.__the_config is None:
            dirname = os.path.dirname(__file__)
            config_full_path = os.path.join(dirname, "..", config_path)
            assert os.path.isfile(config_full_path)
            ConfigProvider.__the_config = ConfigParser(config_full_path).parse()
        return ConfigProvider.__the_config


