""" model """

import json
import logging
import mxnet as mx
from mxnet.model import BatchEndParam
from model.symbol import cnn_dnn_ctc
from model.symbol import cnn_rnn_ctc
from model.symbol import conv2s_ctc
from model.symbol import multi_rnn
from model.symbol import cbhg
from model.metric import SpeechMetric
from model.metric import LanguageMetric
from data.data_loader import SpeechDataLoader
from data.data_loader import LanguageDataLoader


########################################### custom callback ###########################################

class AdaptedLrScheduler():
    """ adapted learning rate scheduler """
    def __init__(self, lr_factor, thresh):
        self.lr_factor = lr_factor
        self.thresh = thresh

    def __call__(self, optimizer, no_improved):
        if no_improved >= self.thresh:
            optimizer.lr = optimizer.lr * self.lr_factor

class BestPerformanceRestore():
    """ restore best performance for model """
    def __init__(self, prefix, criteria, rule):
        assert rule in ['greater', 'less']
        self.prefix = prefix
        self.criteria = criteria
        if rule == 'greater':
            self.is_best = lambda x, y: x > y
        else:
            self.is_best = lambda x, y: x < y
        self.rule = rule

    @property
    def criteria_name(self):
        return self.criteria

    def __call__(self, best_pf, cur_pf, epoch, network):
        cur_val = dict(cur_pf)[self.criteria]
        if self.is_best(cur_val, best_pf['value']):
            mx.model.save_checkpoint(self.prefix, epoch+1, *network)
            best_pf = {'epoch': epoch, 'name': self.criteria, 'value': cur_val}
        return best_pf


########################################### load params ###########################################

def load_params(path):
    """ loading params """
    if path is None:
        return None
    else:
        save_dict = mx.nd.load(path)
        arg_params, aux_params = {}, {}
        for k, v in save_dict.items():
            tp, name = k.split(':', 1)
            if tp == 'arg':
                arg_params[name] = v
            if tp == 'aux':
                aux_params[name] = v
        return {'args': arg_params, 'auxs': aux_params}


########################################### Model ###########################################

class Model():
    """ base model for acoustic model and language model """
    def __init__(self, net_sym, params_path=None):
        self.net_sym = net_sym
        self.net_params = load_params(params_path)

    def _build_metric(self, metric_cfg):
        raise NotImplementedError("please complete this function!")

    def _build_dataiter(self, iter_cfg):
        raise NotImplementedError("please complete this function!")

    def train(self, train_cfg):
        # preparing for training
        if train_cfg['context'][0] == 'gpu':
            context = [mx.gpu(i) for i in train_cfg['context'][1]]
        else:
            context = [mx.cpu(i) for i in train_cfg['context'][1]]
        train_iter = self._build_dataiter(train_cfg['train_iter'])
        valid_iter = self._build_dataiter(train_cfg['valid_iter'])
        # metric
        metric = self._build_metric(train_cfg['metric'])
        # optimizer
        opt_name = train_cfg['optimizer']['name']
        opt_cfg  = train_cfg['optimizer']['config']
        optimizer = mx.optimizer.Optimizer.create_optimizer(opt_name, **opt_cfg)
        # batch_end_callback
        frequent   = train_cfg['batch_end_callback']['frequent']
        batch_size = train_cfg['train_iter']['batch_size']
        batch_end_callback = mx.callback.Speedometer(batch_size, frequent, False)
        # epoch_end_callback: checkpoint
        rule = train_cfg['epoch_end_callback']['rule']
        criteria_name = train_cfg['epoch_end_callback']['criteria_name']
        checkpoint_prefix = train_cfg['epoch_end_callback']['checkpoint_prefix']
        checkpoint = BestPerformanceRestore(checkpoint_prefix, criteria_name, rule)
        # epoch_end_callback: adapted_lr
        lr_factor = train_cfg['epoch_end_callback']['lr_factor']
        thresh    = train_cfg['epoch_end_callback']['thresh']
        adapted_lr = AdaptedLrScheduler(lr_factor, thresh)
        epoch_end_callback = [checkpoint, adapted_lr]
        # training range
        epoch = train_cfg['epoch']
        early_stop = train_cfg['early_stop']

        # model initialization
        data_names = train_cfg['data_names']
        label_names= train_cfg['label_names']
        mod = mx.mod.Module(self.net_sym, data_names, label_names, context=context)

        # training...
        self._train_process(mod, epoch, train_iter, valid_iter,
                metric, optimizer, batch_end_callback, epoch_end_callback, early_stop)

    def eval(self, eval_cfg):
        assert self.net_params is not None, "model params are required!"

        # preparing for evaluating
        if eval_cfg['context'][0] == 'gpu':
            context = [mx.gpu(i) for i in eval_cfg['context'][1]]
        else:
            context = [mx.cpu(i) for i in eval_cfg['context'][1]]
        eval_iter  = self._build_dataiter(eval_cfg['eval_iter'])
        metric     = self._build_metric(eval_cfg['metric'])
        frequent   = eval_cfg['frequent']
        batch_size = eval_cfg['eval_iter']['batch_size']
        callback   = mx.callback.Speedometer(batch_size, frequent, False)

        # model initialization
        out_names  = eval_cfg['out_names']
        data_names = eval_cfg['data_names']
        out_sym = mx.sym.Group([self.net_sym.get_internals()[name] for name in out_names])
        mod = mx.mod.Module(out_sym, data_names=data_names, label_names=[], context=context)  # no label
        mod.bind(data_shapes=eval_iter.provide_data, for_training=False)
        mod.set_params(self.net_params['args'], self.net_params['auxs'], allow_missing=False)

        # evaluating
        criteria, predicts = self._eval_process(mod, eval_iter, metric, callback)
        logging.info('\t'.join(['Evaluattion:'] + ['%s=%f' % (n,v) for n, v in criteria]) + '\n')
        mx.nd.save(eval_cfg['predict_path'], predicts)

    def _eval_process(self, mod, eval_iter, metric, callback):
        outputs = []
        metric.reset()
        eval_iter.reset()
        for nbatch, batch in enumerate(eval_iter):
            # forward
            mod.prepare(batch)
            mod.forward(batch, is_train=False)
            # remove padded samples
            num_pad = batch.pad
            batch_label = batch.label
            batch_out = mod.get_outputs()
            if num_pad > 0:
                batch_out  = [out[:-num_pad] if len(out) > num_pad else out for out in batch_out]  # shorter is loss
                batch_label= [lab[:-num_pad] for lab in batch_label]
            # metric
            metric.update(batch_label, batch_out)
            # callback
            callback(BatchEndParam(0, nbatch, metric, locals()))
            # save outputs
            if nbatch == 0:
                outputs = [bout.copy() for bout in batch_out]
            else:
                outputs = [mx.nd.concat(out, bout, dim=0) for out, bout in zip(outputs, batch_out)]

        criteria = metric.get_name_value()
        return criteria, outputs

    def _train_process(self, mod, epoch, train_iter, valid_iter,
            metric, optimizer, batch_end_callback, epoch_end_callback, early_stop=30):
        # params extracting
        checkpoint, adapted_lr = epoch_end_callback
        arg_params = self.net_params['args'] if self.net_params is not None else None
        aux_params = self.net_params['auxs'] if self.net_params is not None else None
        # model initialization
        mod.bind(train_iter.provide_data, train_iter.provide_label, for_training=True)
        mod.init_params(initializer=mx.init.Xavier(),
                        arg_params=arg_params,
                        aux_params=aux_params,
                        allow_missing=False)
        mod.init_optimizer(optimizer=optimizer)

        # training loops
        if checkpoint.rule == "greater":
            best = {'epoch': 0, 'name': checkpoint.criteria_name, 'value': 0}
        else:
            best = {'epoch': 0, 'name': checkpoint.criteria_name, 'value': 1e10}
        for nepoch in range(epoch):
            metric.reset()
            train_iter = iter(train_iter)
            for nbatch, batch in enumerate(train_iter):
                mod.forward(batch, is_train=True)
                mod.update_metric(metric, batch.label)
                mod.backward()
                mod.update()
                batch_end_callback(BatchEndParam(nepoch, nbatch, metric, locals()))
            # training result present
            result = metric.get_name_value()
            logging_line = 'Epoch[%d]\t' % nepoch
            logging_line += '\t'.join(['Train-%s=%f' % (n, v) for n, v in result])
            logging.info(logging_line)
            # sync aux params across devices
            arg_params, aux_params = mod.get_params()
            mod.set_params(arg_params, aux_params)
            # evaluation
            res, _ = self._eval_process(mod, valid_iter, metric, batch_end_callback)
            # epoch end callback
            best = checkpoint(best, res, nepoch, (mod.symbol, arg_params, aux_params))
            adapted_lr(optimizer, nepoch - best['epoch'])  # adapted learning rate
            # early stop
            logging.info('Validation Best performance: Epoch[%d] %s=%f'
                    % (best['epoch'], best['name'], best['value']))
            if nepoch - best['epoch'] > early_stop:
                break
            # reset train iter
            train_iter.reset()


class AcousticModel(Model):
    def __init__(self, net_info, params_path=None):
        if net_info['net_name'] == 'cnn_dnn_ctc':
            net_sym = cnn_dnn_ctc(net_info['net_cfg'])
        elif net_info['net_name'] == 'cnn_rnn_ctc':
            net_sym = cnn_rnn_ctc(net_info['net_cfg'])
        else:
            raise ValueError('network is not supported!')
        super(AcousticModel, self).__init__(net_sym, params_path)

    def _build_dataiter(self, iter_cfg):
        # extracting params
        list_path = iter_cfg['list_path']
        max_seq = iter_cfg['max_seq']
        max_label = iter_cfg['max_label']
        batch_size = iter_cfg['batch_size']
        shrink_times = iter_cfg['shrink_times']
        shuffle = iter_cfg['shuffle']
        num_workers = iter_cfg['num_workers']
        with open(iter_cfg['py_to_idx_path']) as fd:
            py_to_idx = json.load(fd)

        return SpeechDataLoader(list_path, py_to_idx,
                max_seq, max_label, batch_size,
                shrink_times, shuffle, num_workers)

    def _build_metric(self, metric_cfg):
        return SpeechMetric(metric_cfg['phase'])


class LanguageModel(Model):
    def __init__(self, net_info, params_path):
        if  net_info['net_name'] == 'multi_rnn':
            net_sym = multi_rnn(net_info['net_cfg'])
        elif net_info['net_name'] == 'cbhg':
            net_sym = cbhg(net_info['net_cfg'])
        else:
            raise ValueError('network is not supported!')
        super(LanguageModel, self).__init__(net_sym, params_path)

    def _build_dataiter(self, iter_cfg):
        # extracting params
        list_path = iter_cfg['list_path']
        input_depth = iter_cfg['input_depth']
        seq_len = iter_cfg['seq_len']
        batch_size = iter_cfg['batch_size']
        shuffle = iter_cfg['shuffle']
        num_workers = iter_cfg['num_workers']
        with open(iter_cfg['py_to_idx_path']) as fd:
            py_to_idx = json.load(fd)
        with open(iter_cfg['ch_to_idx_path']) as fd:
            ch_to_idx = json.load(fd)

        return LanguageDataLoader(list_path,
                input_depth, seq_len, py_to_idx,
                ch_to_idx, batch_size, shuffle,
                num_workers)

    def _build_metric(self, metric_cfg):
        return LanguageMetric()
