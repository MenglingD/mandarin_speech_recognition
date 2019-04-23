""" wrapper data_loader """


import mxnet as mx
import numpy as np
from functools import partial


################################################################# acoustic dataloder #################################################################

class SpeechDataSet(mx.gluon.data.dataset.Dataset):
    """ speech dataset """
    def __init__(self, list_path, py_to_idx, shrink_times):
        samples = []
        with open(list_path) as fd:
            for line in fd.readlines():
                _, feat_path, pys_str, _ = line.split('\t')
                pys = pys_str.split(',')
                label = [py_to_idx[py] for py in pys]
                samples.append((feat_path, label))
        self.samples = samples
        self.shrink_times = shrink_times

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = mx.nd.array(np.load(path), dtype='float32')
        data = mx.nd.expand_dims(data, axis=0)
        label = mx.nd.array(label, dtype='float32')
        data_len = len(data[0]) // self.shrink_times
        label_len= len(label)

        return data, label, data_len, label_len

    def __len__(self):
        return len(self.samples)

def _speech_batchify_fn(samples, max_seq, max_label):
    """Collate data into batch for speech data."""
    data = list(map(lambda x: x[0], samples))
    label= list(map(lambda x: x[1], samples))
    data_len = list(map(lambda x: x[2], samples))
    label_len= list(map(lambda x: x[3], samples))
    batch_label= mx.nd.zeros(shape=(len(label), max_label), dtype=label[0].dtype)
    batch_data = mx.nd.zeros(shape=(len(data), 1, max_seq, data[0].shape[-1]), dtype=data[0].dtype)
    for i, data_samp in enumerate(data):  # padding
        batch_data[i, :, :data_samp.shape[1]] =  data_samp
    for i, label_samp in enumerate(label):       # padding
        batch_label[i, :len(label_samp)] = label_samp
    batch_data_len = mx.nd.array(data_len, dtype='float32')
    batch_label_len= mx.nd.array(label_len, dtype='float32')

    return batch_data, batch_label, batch_data_len, batch_label_len


class SpeechDataLoader():
    """ speech dataloader """
    def __init__(self, list_path, py_to_idx, max_seq, max_label,
            batch_size, shrink_times, shuffle=True, num_workers=0):
        dataset = SpeechDataSet(list_path, py_to_idx, shrink_times)
        batchify_fn = partial(_speech_batchify_fn, max_seq=max_seq, max_label=max_label)
        self.data_loader = mx.gluon.data.DataLoader(
                dataset     = dataset,
                batch_size  = batch_size,
                shuffle     = shuffle,
                last_batch  = 'keep',
                num_workers = num_workers,
                batchify_fn = batchify_fn)

        self.batch_size = batch_size
        self.provide_data = [mx.io.DataDesc(**{'name': 'data', 'shape': (batch_size, 1, max_seq, 200)})]
        self.provide_label = [mx.io.DataDesc(**{'name': 'label', 'shape': (batch_size, max_label)}),
                mx.io.DataDesc(**{'name': 'data_len', 'shape': (batch_size,)}),
                mx.io.DataDesc(**{'name': 'label_len', 'shape': (batch_size,)})]
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_iter = iter(self.data_loader)

    def __next__(self):
        return self.next()

    def next(self):
        data, label, data_len, label_len = next(self.data_iter)
        num_pad = self.batch_size - len(data)
        if num_pad > 0: # zero padding for last batch
            # zeros batch
            raw_data, raw_label, raw_data_len, raw_label_len = data, label, data_len, label_len
            data = mx.nd.zeros(shape=(self.batch_size, *data.shape[1:]), dtype=data.dtype)
            label= mx.nd.zeros(shape=(self.batch_size, *label.shape[1:]), dtype=label.dtype)
            data_len = mx.nd.zeros(shape=(self.batch_size, ), dtype=data_len.dtype)
            label_len= mx.nd.zeros(shape=(self.batch_size, ), dtype=label_len.dtype)
            # padding
            data[:-num_pad] = raw_data
            label[:-num_pad] = raw_label
            data_len[:-num_pad] = raw_data_len
            label_len[:-num_pad] = raw_label_len
        labels = [label, data_len, label_len]
        return mx.io.DataBatch(data=[data], label=labels, pad=num_pad)


################################################################# language dataloder #################################################################

class LanguageDataSet(mx.gluon.data.dataset.Dataset):
    """ language dataset """
    def __init__(self, list_path, input_depth, py_to_idx, ch_to_idx):
        self.input_depth = input_depth
        samples = []
        with open(list_path) as fd:
            for line in fd.readlines():
                _, _, pys_str, chs_str = line.split('\t')
                chs_str = chs_str[:-1]  # ignore '\n'
                py_seq = [py_to_idx[py] for py in pys_str.split(',')]
                ch_seq = [ch_to_idx[ch] for ch in chs_str.split(',')]
                samples.append((py_seq, ch_seq))
        self.samples = samples

    def __getitem__(self, idx):
        py_seq, ch_seq = self.samples[idx]
        py_seq = mx.nd.array(py_seq)
        data = mx.nd.one_hot(py_seq, self.input_depth)
        label = mx.nd.array(ch_seq)
        return data, label

    def __len__(self):
        return len(self.samples)


def _language_batchify_fn(samples, seq_len):
    data = list(map(lambda x: x[0], samples))
    label = list(map(lambda x: x[1], samples))
    batch_data = mx.nd.zeros(shape=(len(data), seq_len, data[0].shape[-1]), dtype=data[0].dtype)
    batch_label= mx.nd.zeros(shape=(len(label), seq_len), dtype=label[0].dtype)
    for i, data_samp in enumerate(data):
        batch_data[i, :data_samp.shape[0], :] = data_samp
    for i, label_samp in enumerate(label):
        batch_label[i, :label_samp.shape[0]] = label_samp

    return batch_data, batch_label


class LanguageDataLoader():
    """ language dataloader """
    def __init__(self, list_path, input_depth, seq_len, py_to_idx, ch_to_idx,
            batch_size, shuffle=True, num_workers=0):
        dataset = LanguageDataSet(list_path, input_depth, py_to_idx, ch_to_idx)
        batchify_fn = partial(_language_batchify_fn, seq_len=seq_len)
        self.data_loader = mx.gluon.data.DataLoader(
                dataset     = dataset,
                batch_size  = batch_size,
                shuffle     = shuffle,
                last_batch  = 'keep',
                num_workers = num_workers,
                batchify_fn = batchify_fn)
        self.batch_size = batch_size
        self.provide_data = [mx.io.DataDesc(**{'name': 'data', 'shape': (batch_size, seq_len, input_depth)})]
        self.provide_label = [mx.io.DataDesc(**{'name': 'label', 'shape': (batch_size, seq_len)})]
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        self.data_iter = iter(self.data_loader)

    def __next__(self):
        return self.next()

    def next(self):
        data, label = next(self.data_iter)
        num_pad = self.batch_size - len(data)
        if num_pad > 0: # zero padding for last batch
            # zeros batch
            raw_data, raw_label = data, label
            data = mx.nd.zeros(shape=(self.batch_size, *data.shape[1:]), dtype=data.dtype)
            label= mx.nd.zeros(shape=(self.batch_size, *label.shape[1:]), dtype=label.dtype)
            # padding
            data[:-num_pad], label[:-num_pad] = raw_data, raw_label
        return mx.io.DataBatch(data=[data], label=[label], pad=num_pad)


if __name__ == '__main__':
    import json
    import time

    list_path = '/home/dengmengling/Public/thchs30_fbank/list_files/train.lst'
    py_dict = '/home/dengmengling/Public/thchs30_fbank/dict/pinyins_to_index.json'
    ch_dict = '/home/dengmengling/Public/thchs30_fbank/dict/characters_to_index.json'
    with open(py_dict) as fd:
        py_to_idx = json.load(fd)
    with open(ch_dict) as fd:
        ch_to_dix = json.load(fd)
    py_dict = '/home/dengmengling/Public/thchs30_fbank/dict/index_to_pinyins.json'
    ch_dict = '/home/dengmengling/Public/thchs30_fbank/dict/index_to_characters.json'
    with open(py_dict) as fd:
        idx_to_py = json.load(fd)
    with open(ch_dict) as fd:
        idx_to_ch = json.load(fd)

    data_loader = SpeechDataLoader(list_path, py_to_idx, 1632, 48, 32, 8, shuffle=False)
    start = time.time()
    for i, batch in enumerate(iter(data_loader)):
        for j, data_samp in enumerate(batch.data[0]):
            print('Batch %d, data_sample shape: ' % i, data_samp.shape)
        for j in range(32):
            print('Batch %d, label_samp shape: ' % i, batch.label[0][j].shape)
            print('Batch %d, data_len_samp shape: ' % i, batch.label[1][j].shape)
            print('Batch %d, label_len_samp shape: ' % i, batch.label[2][j].shape)
    print('totol cost: %f' % (time.time() - start))

    data_loader = LanguageDataLoader(list_path, 1208, 48, py_to_idx, ch_to_dix, 32)
    batch = next(data_loader)
    for batch in data_loader:
        for i in range(len(batch.data)):
            py_seq = [idx_to_py[str(int(idx.asnumpy().tolist()[0]))] for idx in mx.nd.argmax(batch.data[0][i], axis=1)]
            ch_seq = [idx_to_ch[str(int(idx.asnumpy().tolist()[0]))] for idx in batch.label[0][i]]
            py_line = ' '.join(py_seq)
            ch_line = ' '.join(ch_seq)
            print('拼音序列:', py_line)
            print('汉子序列:', ch_line)

    #  start = time.time()
    #  for i, batch in enumerate(iter(data_loader)):
    #      print('data_shape: ', batch.data[0].shape)
    #      print('label_shape: ', batch.label[0].shape)
    #      print('data:', batch.data[0])
    #      print('label:', batch.label[0])
    #  print('totol cost: %f' % (time.time() - start))

