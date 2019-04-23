""" learning curve plot """

import argparse
import numpy as np
import matplotlib.pyplot as plt


def parse_value(line, name):
    item = [x for x in line.split('\t') if name in x][0]
    value = item.split('=')[-1]
    value = float(value[:-1])if '\n' in value else float(value)
    return value


def parse_log(log_path, value_name='loss'):
    with open(log_path) as fd:
        lines = fd.readlines()

    i, epoch = 0, 0
    train_values = []
    valid_values = []
    while i != len(lines):
        # training part
        while 'Epoch' in lines[i] and 'Batch' in lines[i]:
            train_values.append(parse_value(lines[i], value_name))
            i += 1
        train_values.append(parse_value(lines[i], value_name))
        i += 1
        # valid part
        while i < len(lines) and 'Epoch[0]' in lines[i] and 'Batch' in lines[i]:
            valid_values.append(parse_value(lines[i], value_name))
            i += 1
        # skip save message
        while i < len(lines) and 'Batch' not in lines[i]:
            i += 1
        epoch += 1

    return epoch, train_values, valid_values


def plot(epoch, train_values, valid_values=None, name='loss'):
    train_x = np.linspace(0, epoch, len(train_values))
    plt.plot(train_x, train_values, color='blue', label='train_%s' % name)
    if valid_values is not None:
        valid_x = np.linspace(0, epoch, len(valid_values))
        plt.plot(valid_x, valid_values, color='red', label='valid_%s' % name)
    plt.legend()
    plt.xlabel('iteration times')
    plt.ylabel(name)
    plt.xlim(0, epoch)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference for speech recognition')
    parser.add_argument('log_path', type=str, help='wav file path')
    parser.add_argument('--value_name', type=str, help='name of value expected', default='loss')
    args = parser.parse_args()

    epoch, train_values, valid_values = parse_log(args.log_path, args.value_name)
    plot(epoch, train_values, valid_values, args.value_name)
