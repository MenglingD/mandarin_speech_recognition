""" 语言模型标签处理 """

import math
import numpy as np
from functools import reduce
from itertools import groupby


def build_dict(samples):
    """ 生成转换字典:

    py_to_idx: 拼音转序号
    ch_to_idx: 汉字转序号
    idx_to_py: 序号转拼音
    idx_to_ch: 汉字转序号
    """
    pinyins, characters = [], []
    for sample in samples:
        pinyins.append(sample['pinyins'])
        characters.append(sample['characters'])
    pinyins    = sorted(reduce(lambda x, y: set(x) | set(y), pinyins))
    characters = sorted(reduce(lambda x, y: set(x) | set(y), characters))
    pinyins   = ['_'] + list(pinyins)
    py_to_idx = {py: idx for py, idx in zip(pinyins, range(len(pinyins)))}
    ch_to_idx = {ch: idx for ch, idx in zip(characters, range(len(characters)))}
    idx_to_py = {idx: py for py, idx in zip(pinyins, range(len(pinyins)))}
    idx_to_ch = {idx: ch for ch, idx in zip(characters, range(len(characters)))}

    return py_to_idx, ch_to_idx, idx_to_py, idx_to_ch


############################################## probilities decode ##############################################

def beam_search_decode(probs, topK=1):
    """ beam search decode
    :param probs: np.array
        matrix-like with shape: [time_step, num_cls]
    :param topK: int
        top k path
    :return pred: list of list
        top k path prediction
    """
    eps = 1e-12
    sequences = [[list(), 1.0]]
    # loop in time step
    for prob in probs:
        candidates = []
        # expand each current candidate
        for seq, score in sequences:
            # loop for all probilities in a row
            for i, c_prob in enumerate(prob):
                c_prob = max(c_prob, eps)
                candidate = [seq+[i], score*-math.log(c_prob)]
                candidates.append(candidate)
            ordered = sorted(candidates, key=lambda x: x[1])
            sequences = ordered[:topK]
    return sequences


def greedy_decode(probs):
    """ greedy decode
    :param probs: list of list
        matrix-like with shape: [time_step, num_cls]
    :return pred: list with shape [time_step, ]
        decoded path prediction
    """
    return np.argmax(probs, axis=1)


def ctc_alignment(pred):
    """ ctc alignment: reduplicate, remove blank """
    # reduplication
    reduped = np.array([x[0] for x in groupby(pred)])
    # remove blank
    removed = reduped[np.nonzero(reduped)]
    return removed


if __name__ == '__main__':
    # ctc alignment
    pred = np.array([0, 1, 2, 1, 1, 4, 3, 0, 3, 0, 3, 3, 4, 5, 6])
    print('ctc alignment:', ctc_alignment(pred))

    # greedy decode
    probs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5],
	[0.5, 0.4, 0.3, 0.2, 0.1],
	[0.1, 0.2, 0.3, 0.4, 0.5],
	[0.5, 0.4, 0.3, 0.2, 0.1],
	[0.1, 0.2, 0.3, 0.4, 0.5],
	[0.5, 0.4, 0.3, 0.2, 0.1],
	[0.1, 0.2, 0.3, 0.4, 0.5],
	[0.5, 0.4, 0.3, 0.2, 0.1]])
    print('greedy decode:', greedy_decode(probs))

    # beam search decode
    for path in beam_search_decode(probs, 3):
        print('beam search path:', path)
