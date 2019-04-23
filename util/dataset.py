""" dataset utilities(考虑到代码简洁性，未使用并发进行加速) """

import os
import json
import numpy as np
from tqdm import tqdm
from util.vocal import build_dict
from util.audio import compute_fbank


class SpeechDataSet():
    """ base class on madarin speech dataset """
    def __init__(self, name, data_dir, saved_dir):
        """ samples: [{'wav': xx, 'feat': xx, 'pinyins': xx, 'characters': xx}, ...] """
        self.name = name
        self.data_dir = data_dir
        self.saved_dir = saved_dir

    def build_dataset(self):
        """ unification for dataset """
        # building samples
        print('building samples...')
        self.samples = self._build_samples()
        # building dict
        print('building dicts...')
        p2i, c2i, i2p, i2c = build_dict(self.samples)
        dict_dir = os.path.join(self.saved_dir, 'dict')
        os.makedirs(dict_dir)
        with open(os.path.join(dict_dir, 'pinyins_to_index.json'), 'w') as fd:
            json.dump(p2i, fd, ensure_ascii=False, indent=2)
        with open(os.path.join(dict_dir, 'index_to_pinyins.json'), 'w') as fd:
            json.dump(i2p, fd, ensure_ascii=False, indent=2)
        with open(os.path.join(dict_dir, 'characters_to_index.json'), 'w') as fd:
            json.dump(c2i, fd, ensure_ascii=False, indent=2)
        with open(os.path.join(dict_dir, 'index_to_characters.json'), 'w') as fd:
            json.dump(i2c, fd, ensure_ascii=False, indent=2)
        # feature extracting and save
        print('extracting feature...')
        for sample in tqdm(self.samples):
            wav_path = sample['wav']
            feat = compute_fbank(wav_path)
            # restoring
            feat_name = wav_path.split('/')[-1].split('.')[0] + '.npy'
            feat_path = os.path.join(self.saved_dir, feat_name)
            np.save(feat_path, feat)
            # append feature path
            sample['feat'] = feat_path
        # build list for dataset
        print('building lists...')
        self._build_list()

    def _build_samples(self):
        raise NotImplementedError("please complete this!")

    def _build_list(self):
        raise NotImplementedError("please complete this!")
