""" metric utility for acoustic model and language model """

import sys
sys.path.append('../')

import mxnet as mx
import numpy as np
from util.vocal import greedy_decode
from util.vocal import ctc_alignment
from util.vocal import beam_search_decode


################################################################# acoustic metric #################################################################

def _word_error(hyp, ref):
    """ word error count: substitution, insertion, deletion
        reference: https://martin-thoma.com/word-error-rate-calculation
    """
    # initialisation
    r_len, h_len = len(ref), len(hyp)
    d = np.zeros(shape=(r_len+1, h_len+1), dtype=np.uint8)
    for i in range(r_len+1):
        for j in range(h_len+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # computation
    for i in range(1, r_len+1):
        for j in range(1, h_len+1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[r_len][h_len]


class SpeechMetric(mx.metric.EvalMetric):
    def __init__(self, phase):
        assert phase in ['train', 'valid']
        self.phase = phase
        if self.phase == 'train':
            self.names = ['wer', 'ser', 'loss']
        else:
            self.names = ['wer', 'ser']

        self.reset()
        super(SpeechMetric, self).__init__('speech_metric')

    def reset(self):
        self.word_error = 0.0
        self.total_word = 0.0
        self.sent_error = 0.0
        self.total_sent = 0.0
        if self.phase == 'train':
            self.total_loss = 0.0
            self.loss_count = 0.0

    def update(self, labels, preds):
        if self.phase == 'train':
            probs, losses = preds            # probs, loss
            self.total_loss += mx.nd.sum(losses).asscalar()
            self.loss_count += len(losses)
        else:
            probs = preds[0]                 # probs
        labels, data_lens, label_lens = labels   # label, data_len, label_len
        # word error, sentence error
        probs = probs.asnumpy()
        labels = labels.asnumpy()
        data_lens = data_lens.asnumpy().tolist()
        label_lens = label_lens.asnumpy().tolist()
        for prob, label, data_len, label_len in zip(
                probs, labels, data_lens, label_lens):
            # true label
            prob = prob[:int(data_len)]
            label = label[:int(label_len)]
            # decode ctc: [n, s, c] --> [n, c]
            # pred = greedy_decode(prob)
            pred = beam_search_decode(prob, topK=2)[0][0]
            pred = ctc_alignment(pred)
            # word error
            self.total_word += label_len
            self.word_error += _word_error(pred, label)
            # sentence error
            self.total_sent += 1
            if len(pred) == len(label) and sum(pred == label) == len(pred):
                self.sent_error += 0
            else:
                self.sent_error += 1

    def get(self):
        eps = 1e-12
        wer = self.word_error / max(self.total_word, eps)
        ser = self.sent_error / max(self.total_sent, eps)
        if self.phase == 'valid':
            return self.names, [wer, ser]
        else:
            loss = self.total_loss / max(self.loss_count, eps)
            return self.names, [wer, ser, loss]


################################################################# language metric #################################################################

class LanguageMetric(mx.metric.EvalMetric):
    """ language metric """
    def __init__(self, phase='train'):
        self.eps = 1e-12
        self.phase = phase
        self.names = ['acc'] if phase == 'valid' else ['acc', 'loss']
        self.reset()
        super(LanguageMetric, self).__init__('lanuage_metric')

    def reset(self):
        if self.phase == 'train':
            self.loss, self.loss_count = 0.0, 0.0
        self.acc, self.acc_count = 0.0, 0.0

    def update(self, labels, preds):
        label = labels[0]
        if self.phase == 'train':
            probs, loss = preds
            self.loss_count += 1
            self.loss += loss.asscalar()
        else:
            probs = preds[0]
        probs = preds[0].as_in_context(mx.cpu())
        pred = mx.nd.argmax(probs, axis=2)
        num_acc = (pred == label).sum().asscalar()
        self.acc += float(num_acc) / max(float(pred.size), self.eps)
        self.acc_count += 1

    def get(self):
        acc = self.acc / max(float(self.acc_count), self.eps)
        if self.phase == 'train':
            loss = self.loss / max(self.loss_count, self.eps)
            values = [acc, loss]
        else:
            values = [acc]
        return self.names, values


if __name__ == '__main__':
    label = mx.nd.array([[2, 1],
                         [2, 1],
                         [2, 1]])
    prob1 = [[0.2, 0.2, 0.4, 0.2],
             [0.2, 0.3, 0.2, 0.3]]
    preds = [mx.nd.array([prob1, prob1, prob1]), mx.nd.array([0.4, 0.5, 0.6])]
    labels = [label, mx.nd.array([2, 2, 2]), mx.nd.array([2, 2, 2])]
    metric = SpeechMetric('train')
    metric.update(labels, preds)
    for name, value in metric.get_name_value():
        print('%s = %f' % (name, value))

    labels = [label]
    preds = [preds[0], mx.nd.array([0.4, 0.5, 0.6])]
    metric = LanguageMetric('train')
    metric.update(labels, preds)
    for name, value in metric.get_name_value():
        print('%s = %f' % (name, value))
