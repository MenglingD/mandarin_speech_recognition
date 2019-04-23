""" scritpt for training """

import sys
import yaml
from util.log import set_logger
from model.model import AcousticModel
from model.model import LanguageModel


if __name__ == '__main__':
    assert len(sys.argv) == 2, 'config file path is required!'

    with open(sys.argv[1]) as fd:
        cfg = yaml.load(fd, Loader=yaml.SafeLoader)
    # set logger
    set_logger(cfg['logger']['path'],
               cfg['logger']['level'])
    # loading params
    if cfg['model_type'] == 'acoustic':
        model = AcousticModel(cfg['net_info'], cfg['params_path'])
    elif cfg['model_type'] == 'language':
        model = LanguageModel(cfg['net_info'], cfg['params_path'])
    else:
        raise ValueError('model type: [acoustic|language]')
    # model training
    model.eval(cfg['eval_cfg'])
