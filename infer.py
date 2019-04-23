""" script for inference """

import json
import argparse
import mxnet as mx
import numpy as np
from util.audio import compute_fbank
from util.vocal import greedy_decode
from util.vocal import ctc_alignment


def infer(sym, arg_params, aux_params, input, data_shapes):
    # model initialization
    mod = mx.module.Module(sym, data_names=['data'], label_names=[], context=mx.cpu())
    mod.bind(data_shapes=data_shapes, for_training=False)
    mod.set_params(arg_params=arg_params, aux_params=aux_params)
    # prediction
    mod.forward(mx.io.DataBatch(data=[input]), is_train=False)
    prediction = mod.get_outputs()[0].asnumpy()

    return prediction


def speech_recognition(wav, acoustic_model, language_model, idx_to_py, idx_to_ch):
    # acoustic inference
    ast_input = mx.nd.array(compute_fbank(wav))
    ast_data = mx.nd.zeros(shape=(1, 1, 1632, 200), dtype='float32')
    ast_data[:, :, :len(ast_input), :] = ast_input
    ast_data_shapes = [mx.io.DataDesc(**{'name': 'data', 'shape': ast_data.shape})]
    ast_prediction = infer(*acoustic_model, ast_data, ast_data_shapes)
    ast_idx_seq = ctc_alignment(greedy_decode(ast_prediction[0]))  # one sample only
    py_seq = [idx_to_py[str(int(x))] for x in ast_idx_seq]
    print('预测的拼音序列为:')
    print(' '.join(py_seq))

    # language inference
    lng_data = mx.nd.zeros(shape=(1, 48, 1208), dtype='float32')
    py_idx_seq = mx.nd.one_hot(mx.nd.array(ast_idx_seq), depth=1208, dtype='float32')
    seq_len = min(48, py_idx_seq.shape[0])
    lng_data[:, :seq_len, :] = py_idx_seq[:seq_len]
    lng_data_shapes = [mx.io.DataDesc(**{'name': 'data', 'shape': lng_data.shape})]
    lng_prediction = infer(*language_model, lng_data, lng_data_shapes)
    lng_idx_seq = np.argmax(lng_prediction[0], axis=1)[:seq_len]
    ch_seq = [idx_to_ch[str(int(x))] for x in lng_idx_seq]
    print('预测的汉字序列为(语言模型预测最长序列长度为48，声学模型超过部分将被截取):')
    print(' '.join(ch_seq))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference for speech recognition')
    parser.add_argument('wav', type=str, help='wav file path')
    parser.add_argument('--idx_to_py', type=str,
            help='path of dict for convertion from index to pinyin',
            default='/home/dengmengling/Public/thchs30_fbank/dict/index_to_pinyins.json')
    parser.add_argument('--idx_to_ch', type=str,
            help='path of dict for convertion from index to characters',
            default='/home/dengmengling/Public/thchs30_fbank/dict/index_to_characters.json')
    parser.add_argument('--acoustic_checkpoint', type=str,
            help='symbol path of acoustic model',
            default='/home/dengmengling/Downloads/cnn_rnn_ctc/cnn_rnn_ctc')
    parser.add_argument('--acoustic_epoch', type=int, help='acoustic checkpoint epoch', default=196)
    parser.add_argument('--acoustic_out_name', type=str, help='name of acoustic output layer', default='predict_output')
    parser.add_argument('--language_checkpoint', type=str,
            help='symbol path of acoustic model',
            default='/home/dengmengling/Downloads/lng_cbhg/lng_cbhg')
    parser.add_argument('--language_epoch', type=int, help='language checkpoint epoch', default=14)
    parser.add_argument('--language_out_name', type=str, help='name of language output layer', default='predict_output')
    args = parser.parse_args()

    # args parsing
    with open(args.idx_to_py) as fd:
        idx_to_py = json.load(fd)
    with open(args.idx_to_ch) as fd:
        idx_to_ch = json.load(fd)
    ast_sym, ast_arg_params, ast_aux_params = mx.model.load_checkpoint(args.acoustic_checkpoint, args.acoustic_epoch)
    lng_sym, lng_arg_params, lng_aux_params = mx.model.load_checkpoint(args.language_checkpoint, args.language_epoch)
    ast_sym, lng_sym = ast_sym.get_internals()[args.acoustic_out_name], lng_sym.get_internals()[args.language_out_name]

    speech_recognition(args.wav,
            (ast_sym, ast_arg_params, ast_aux_params),
            (lng_sym, lng_arg_params, lng_aux_params),
            idx_to_py, idx_to_ch)
