""" dataset preprocess tool """
# parent path
import sys
sys.path.append('../')

import os
from util.dataset import SpeechDataSet


class THCHS30(SpeechDataSet):
    """ thchs30 dataset """
    def __init__(self, data_dir, saved_dir):
        super(THCHS30, self).__init__('THCHS30', data_dir, saved_dir)

    def _build_samples(self):
        """ implement super function """
        data_dir  = os.path.join(self.data_dir, 'data_thchs30', 'data')
        train_dir = os.path.join(self.data_dir, 'data_thchs30', 'train')
        valid_dir = os.path.join(self.data_dir, 'data_thchs30', 'dev')
        test_dir  = os.path.join(self.data_dir, 'data_thchs30', 'test')
        # partation list
        train_wav_paths, train_sct_paths = THCHS30._extract_paths(train_dir, data_dir)
        valid_wav_paths, valid_sct_paths = THCHS30._extract_paths(valid_dir, data_dir)
        test_wav_paths,  test_sct_paths  = THCHS30._extract_paths(test_dir, data_dir)
        self.train_samples = THCHS30._extract_samples(train_wav_paths, train_sct_paths, self.saved_dir)
        self.valid_samples = THCHS30._extract_samples(valid_wav_paths, valid_sct_paths, self.saved_dir)
        self.test_samples  = THCHS30._extract_samples(test_wav_paths, test_sct_paths, self.saved_dir)
        # total samples
        wav_paths, sct_paths = THCHS30._extract_paths(data_dir, data_dir)
        return THCHS30._extract_samples(wav_paths, sct_paths, self.saved_dir)

    def _build_list(self):
        """ implement super function """
        list_dir = os.path.join(self.saved_dir, 'list_files')
        if not os.path.exists(list_dir):
            os.makedirs(list_dir)
        # writing samples to file
        THCHS30._restore_samples(self.train_samples, os.path.join(list_dir, 'train.lst'))
        THCHS30._restore_samples(self.valid_samples, os.path.join(list_dir, 'valid.lst'))
        THCHS30._restore_samples(self.test_samples, os.path.join(list_dir, 'test.lst'))

    @staticmethod
    def _extract_paths(link_dir, raw_dir):
        """ reflect wav path and script to really path """
        wav_names = list(filter(lambda f: f.endswith('.wav'), os.listdir(link_dir)))
        sct_names = [name + '.trn' for name in wav_names]
        wav_paths = [os.path.join(raw_dir, name) for name in wav_names]
        sct_paths = [os.path.join(raw_dir, name) for name in sct_names]
        return wav_paths, sct_paths

    @staticmethod
    def _extract_samples(wav_paths, sct_paths, feat_dir):
        # internal function for concurrency
        samples = []
        for wav_path, sct_path in zip(wav_paths, sct_paths):
            with open(sct_path) as fd:
                chs = [ch for ch in ''.join(fd.readline()[:-1].split(' '))]  # list of per character
                pys = fd.readline()[:-1].split(' ')                          # list of per pinyin
                feat_path = os.path.join(feat_dir, wav_path.split('/')[-1].split('.')[0]+'.npy')
                sample = {'wav': wav_path, 'feat': feat_path, 'pinyins': pys, 'characters': chs}
            samples.append(sample)
        return samples

    @staticmethod
    def _restore_samples(samples, saved_path):
        with open(saved_path, 'w') as fd:
            for sample in samples:
                wav  = sample['wav']
                feat = sample['feat']
                py_str = ','.join(sample['pinyins'])
                ch_str = ','.join(sample['characters'])
                fd.write('\t'.join([wav, feat, py_str, ch_str]) + '\n')


if __name__ == '__main__':
    thshs30 = THCHS30('/home/dml/speech_recognition/dataset', '/home/dml/speech_recognition/dataset/fbank_thchs30_tmp')
    thshs30.build_dataset()
