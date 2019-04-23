""" raw audio process utils which consists of feature extract, and so on """

import numpy as np
from scipy.fftpack import fft
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from python_speech_features import mfcc


def compute_mfcc(file):
    """ 获取mfcc特征 """
    # loading
    fs, audio = wav.read(file)
    # get mfcc feature
    feature = mfcc(audio, samplerate=fs, numcep=26)
    feature = feature[::3]

    return np.transpose(feature)


def compute_fbank(file):
    """ 获取信号时频图 """
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, audio = wav.read(file)
    # 加窗以及时移10ms
    window_ms = 25  # 单位ms
    wav_arr = np.array(audio)
    num_window = int(len(audio) / fs*1000 - window_ms) // 10 + 1  # 最终的窗口数
    fbank = np.zeros((num_window, 200), dtype=np.float)
    for i in range(0, num_window):
        section = np.array(wav_arr[i*160:(i*160)+400])  # 分帧
        section = section * w  # 加窗
        section = np.abs(fft(section))  # 傅里叶变化
        fbank[i] = section[:200]  # 取一半
    fbank = np.log(fbank + 1)  # 取对数，求db

    return fbank


if __name__ == "__main__":
    audio_path = 'D8_999.wav'
    fs, audio = wav.read(audio_path)

    # 声波图
    plt.plot(audio)
    plt.show()

    # 音频的时频图
    fbank = compute_fbank(audio_path)
    plt.imshow(fbank.T, origin='lower')
    plt.show()
