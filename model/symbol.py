""" network symbol """

import mxnet as mx


####################################### acoustic symbol #######################################

def cnn_dnn_ctc(net_cfg):
    # extracting params
    num_cls = net_cfg['num_cls']
    feat_dim = net_cfg['feat_dim']

    # symbol building
    x = _conv_sequence(feat_dim)  # output shape: (batch_size, time_step, feat_dim)
    x = mx.sym.FullyConnected(x, num_hidden=num_cls, flatten=False)
    ctc_output = _ctc_block(x)
    return ctc_output


def cnn_rnn_ctc(net_cfg):
    # extracting params
    seq_len = net_cfg['seq_len']
    feat_dim = net_cfg['feat_dim']
    cell_type = net_cfg['cell_type']
    num_hiddens = net_cfg['num_hiddens']
    num_cls = net_cfg['num_cls']

    # symbol building
    x = _conv_sequence(feat_dim)
    rnn_output, _ = _multi_rnn(x, seq_len, cell_type, num_hiddens)  # ignore state
    fc = mx.sym.FullyConnected(rnn_output, num_hidden=num_cls, flatten=False)
    ctc_output = _ctc_block(fc)
    return ctc_output


def conv2s_ctc(net_cfg):
    pass # TODO: xxx


####################################### language symbol #######################################

def multi_rnn(net_cfg):
    # extracting params
    seq_len = net_cfg['seq_len']
    cell_type = net_cfg['cell_type']
    num_hiddens = net_cfg['num_hiddens']
    output_cls = net_cfg['output_cls']

    data = mx.sym.Variable('data')  # shape: [batch_size, time_step, embedding]
    label = mx.sym.Variable('label') # one-hot shape: [batch_size, time_step]
    rnn_output, _ = _multi_rnn(data, seq_len, cell_type, num_hiddens)
    # output block
    fc = mx.sym.FullyConnected(rnn_output, num_hidden=output_cls, flatten=False)
    predict, loss = _softmax_block(fc, output_cls, prefix='softmax_block')

    return mx.sym.Group([predict, loss])


def cbhg(net_cfg):
    # extracting params
    seq_len = net_cfg['seq_len']
    input_cls = net_cfg['input_cls']
    output_cls = net_cfg['output_cls']

    data = mx.sym.Variable('data')  # shape: [N, T, C]
    label = mx.sym.Variable('label') # shape: [N, T]
    # prenet
    x = mx.sym.FullyConnected(data, num_hidden=input_cls, flatten=False)
    x = mx.sym.Activation(x, act_type='relu')
    x = mx.sym.Dropout(x, p=0.5)
    x = mx.sym.FullyConnected(x, num_hidden=input_cls//2, flatten=False)
    x = mx.sym.Activation(x, act_type='relu')
    x = mx.sym.Dropout(x, p=0.5)
    x = mx.sym.expand_dims(x, axis=1)  # [N, T, C] --> [N, 1, T, C]
    pre_out = mx.sym.swapaxes(x, 1, 3)       # [N, 1, T, C] --> [N, C, T, 1]
    # conv1d bank
    x = _conv1d_blank(pre_out, input_cls//2, 16) # output shape: [N, C*k, T, 1]
    # max pooling for 1d
    x = _pooling1d(x, 2, prefix='max_pooling1d')
    # conv1d projections
    x = _conv1d(x, input_cls//2, 5, prefix='conv1d_pro_1')  # input: [N, C*k, T, 1] output: [N, C, T, 1]
    x = mx.sym.BatchNorm(x)
    x = _conv1d(x, input_cls//2, 5, prefix='conv1d_pro_2')
    x = mx.sym.BatchNorm(x)
    x = x + pre_out  # residual connections [N, C, T, 1]
    x = mx.sym.swapaxes(x, 1, 3)  # [N, C, T, 1] --> [N, 1, T, C]
    x = mx.sym.squeeze(x, axis=1) # [N, 1, T, C] --> [N, T, C]
    # highway block
    for i in range(4):
        x = _highway_block(x, input_cls//2, 'highway_%2d' % (i+1))
    # bidirectional gru
    rnn_output, _ = _multi_rnn(x, seq_len, 'bigru', [input_cls//2])
    # output layer
    fc = mx.sym.FullyConnected(rnn_output, num_hidden=output_cls, flatten=False)
    predict, loss = _softmax_block(fc, output_cls, prefix='softmax_block')

    return mx.sym.Group([predict, loss])


####################################### symbol modules #######################################

def _highway_block(input, num_hidden, prefix=''):
    """ highway block
    :params input: mx.symbol
        with shape [N, T, C]
    :params num_hidden: integer
        number of hidden cells
    """
    with mx.name.Prefix(prefix):
        H = mx.sym.FullyConnected(input, num_hidden=num_hidden, flatten=False)
        H = mx.sym.Activation(H, act_type='relu')
        T = mx.sym.FullyConnected(input, num_hidden=num_hidden, flatten=False)
        T = mx.sym.Activation(T, act_type='sigmoid')
        output = H * T + input * (1.0 - T)

    return output


def _pooling1d(input, k, prefix=''):
    """ wrapper for 1d max pooling with same padding """
    with mx.name.Prefix(prefix):
        left = (k-1) // 2
        right = left if (k-1) % 2 == 0 else left + 1
        x = mx.sym.pad(input, mode='constant', pad_width=(0, 0, 0, 0, left, right, 0, 0))  # zero padding
        x = mx.sym.Pooling(x, kernel=(k, 1), pool_type='max') # no more padding
    return x


def _conv1d(input, num_filter, k, prefix=''):
    """ conv1d wrapper for same padding
    :params input: mx.symbol
        input with shape [N, C, T, 1]
    :params num_filter: integer
        number of filter
    :params k: integer
        kernel size
    """
    with mx.name.Prefix(prefix):
        left = (k-1) // 2
        right = left if (k-1) % 2 == 0 else left + 1
        x = mx.sym.pad(input, mode='constant', pad_width=(0, 0, 0, 0, left, right, 0, 0))  # zero padding
        x = mx.sym.Convolution(x, kernel=(k, 1), num_filter=num_filter) # no more padding
    return x


def _conv1d_blank(input, num_filter, K):
    """ conv1d_blank for cbhg
    :params input: mx.symbol
        input with shape: [N, C, T, 1]
    :params K integer
        number of layers
    :params num_filter:
        number of filter
    """
    output = _conv1d(input, num_filter, 1, 'conv1d_blank_01k')
    for k in range(2, K+1):
        x = _conv1d(input, num_filter, k, 'conv1d_blank_%02dk' % k)  # [N, C, T, 1]
        output = mx.sym.concat(output, x, dim=1) # [N, C*(k-1), T, 1] and [N, C, T, 1] --> [N, C*k, T, 1]
    output = mx.sym.BatchNorm(output)
    return output


def _conv_block(input, num_filter, use_pool=True):
    x = mx.sym.Convolution(input, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, no_bias=False)
    x = mx.sym.BatchNorm(x)
    x = mx.sym.Convolution(x, kernel=(3, 3), pad=(1, 1), num_filter=num_filter, no_bias=False)
    x = mx.sym.BatchNorm(x)
    if use_pool:
        x = mx.sym.Pooling(x, (2, 2), 'max', stride=(2, 2))
    return x


def _conv_sequence(feat_dim):
    data = mx.sym.Variable(name='data')
    x = _conv_block(data, 32)
    x = _conv_block(x, 64)
    x = _conv_block(x, 128)
    x = _conv_block(x, 128, use_pool=False)
    x = _conv_block(x, 128, use_pool=False)
    # reshape and dense
    x = mx.sym.transpose(x, axes=(0, 2, 3, 1))  # (batch_size, num_filter, seq_len, feat_size) --> (batch_size, seq_len, feat_size, num_filter)
    x = mx.sym.reshape(x, shape=(0, 0, -3))     # (batch_size, seq_len, feat_size, num_filter) --> (batch_size, seq_len, feat_size * num_filter)
    x = mx.sym.Dropout(x, p=0.2)
    x = mx.sym.FullyConnected(x, num_hidden=feat_dim, flatten=False)
    x = mx.sym.Dropout(x, p=0.2)
    return x


def _ctc_block(fc):
    label = mx.sym.Variable('label')         # shape: (batch_size, max_label_len)
    data_len = mx.sym.Variable('data_len')   # shape: (batch_size,)
    label_len = mx.sym.Variable('label_len') # shape: (batch_size,)
    # swapaxis (batch_size, max_seq_len, num_cls) --> (max_seq_len, batch_size, num_cls)
    data = mx.sym.swapaxes(fc, dim1=0, dim2=1)
    # loss
    ctc_loss = mx.sym.ctc_loss(data, label, data_len, label_len, True, True)
    ctc_loss = mx.sym.MakeLoss(ctc_loss, name='ctc_make_loss')
    # output
    predict = mx.sym.softmax(fc, axis=2)
    predict = mx.sym.BlockGrad(mx.sym.MakeLoss(predict), name='predict')
    return mx.sym.Group([predict, ctc_loss])


def _softmax_block(fc, out_cls, prefix='softmax_block'):
    """ softmax block with reture prediction and loss """
    eps = 1e-12
    label = mx.sym.Variable('label')
    with mx.name.Prefix(prefix):
        probs = mx.sym.softmax(fc, axis=2)  # with shape [N, T, C]
        # loss
        pos_log = mx.sym.log(mx.sym.clip(probs, eps, 1.0))
        neg_log = mx.sym.log(mx.sym.clip(1-probs, eps, 1.0))
        label = mx.sym.one_hot(label, depth=out_cls) # one-hot shape [N, T] --> [N, T, C]
        loss = - (label * pos_log + (1-label) * neg_log)
        loss = mx.sym.mean(mx.sym.sum(loss, axis=[1, 2]))
    loss = mx.sym.MakeLoss(loss, name='loss')
    predict = mx.sym.BlockGrad(mx.sym.MakeLoss(probs), name='predict')

    return mx.sym.Group([predict, loss])


def _multi_rnn(inputs, seq_len, cell_type, num_hiddens, prefix='rnn_layer'):
    """ multi layer of rnn cell, input shape: (NTC) """
    if cell_type == 'lstm':
        rnn_cells = [mx.rnn.LSTMCell(num_hid, prefix=prefix+'_%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
    elif cell_type == 'gru':
        rnn_cells = [mx.rnn.GRUCell(num_hid, prefix=prefix+'_%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
    elif cell_type == 'bilstm':
        l_cells = [mx.rnn.LSTMCell(num_hid, prefix=prefix+'_l%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
        r_cells = [mx.rnn.LSTMCell(num_hid, prefix=prefix+'_r%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
        rnn_cells = [mx.rnn.BidirectionalCell(l, r, output_prefix=prefix+'_bi%02d_' % i)
                for i, (l, r) in enumerate(zip(l_cells, r_cells))]
    elif cell_type == 'bigru':
        l_cells = [mx.rnn.GRUCell(num_hid, prefix=prefix+'_l%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
        r_cells = [mx.rnn.GRUCell(num_hid, prefix=prefix+'_r%02d_' % i)
                for i, num_hid in enumerate(num_hiddens)]
        rnn_cells = [mx.rnn.BidirectionalCell(l, r, output_prefix=prefix+'_bi%02d_' % i)
                for i, (l, r) in enumerate(zip(l_cells, r_cells))]
    else:
        raise ValueError('wrong cell type![lstm|gru|bilstm|bigru]')

    rnn_layers = mx.rnn.SequentialRNNCell()
    for rnn_cell in rnn_cells:
        rnn_layers.add(rnn_cell)
    outputs, state = rnn_layers.unroll(seq_len, inputs, merge_outputs=True)
    return outputs, state


class LSTMAttentionCell(mx.rnn.rnn_cell.LSTMCell):
    """ LSTM cell with attention mechanism """
    def __init__(self, num_hidden, prefix):
        super(LSTMAttentionCell, self).__init__(num_hidden, prefix)

    def unroll(self, length, inputs):
        pass # TODO: other params set default


class GRUAttentionCell(mx.rnn.rnn_cell.GRUCell):
    """ GRU cell with attention mechanism """
    def __init__(self, num_hidden, prefix):
        super(LSTMAttentionCell, self).__init__(num_hidden, prefix)

    def unroll(self, length, inputs):
        pass # TODO: other params set default


if __name__ == '__main__':
    #  data = mx.sym.Variable('data')
    #  net_cfg = {'input_cls': 12, 'output_cls': 28, 'seq_len': 4}
    #  shape = {'data': (32, 4, 12), 'label': (32, 4)}
    #  output = cbhg(net_cfg)
    #  mx.viz.plot_network(output, shape=shape).view()
    #  print(output.infer_shape(**shape)[1])

    #  shape = {'data': (32, 1, 1632, 200),
    #           'label':(32, 48),
    #           'data_len': (32,),
    #           'label_len': (32,)}
    #  net_cfg = {'feat_dim': 256, 'num_cls': 1208}
    #  output = cnn_dnn_ctc(net_cfg)
    #  print(output.infer_shape(**shape)[1])
    #
    #  shape = {'data': (32, 1, 1632, 200), 'label': (32, 48), 'data_len': (32,), 'label_len': (32,)}
    #  net_cfg = {'feat_dim': 256, 'seq_len': 204, 'num_hiddens': [128, 256, 1208], 'cell_type': 'lstm'}
    #  output = cnn_rnn_ctc(net_cfg)
    #  print(output.infer_shape(**shape)[1])

    shape = {'data': (32, 5, 12), 'label':(32, 5)}
    net_cfg = {'seq_len': 5, 'num_hiddens': [10], 'cell_type': 'lstm', 'output_cls': 28}
    output = multi_rnn(net_cfg)
    mx.viz.plot_network(output, shape=shape).view()
