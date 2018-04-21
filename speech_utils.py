import glob as glob
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import signal
from scipy.io import wavfile


labels = 'silence unknown'
# labels = 'yes no up down left right on off stop go silence unknown'
POSSIBLE_LABELS = set(labels.split())
id2word = {i: word for i, word in enumerate(POSSIBLE_LABELS)}
word2id = {word: i for i, word in id2word.items()}


def load_wavfile(file_path):
  sample_rate, audio = wavfile.read(file_path)
  return sample_rate, audio

def save_wavfile(path, file_name, sample_rate, audio):
  wavfile.write(path + file_name, sample_rate, audio)

#Credit
# https://www.kaggle.com/davids1992/speech-representation-and-data-exploration
def log_spectrogram(audio, sample_rate, window_sz=20, step_sz=10, eps=1e-10):
  nperseg = int(np.around(window_sz * sample_rate / 1e3))
  noverlap = int(np.around(step_sz * sample_rate / 1e3))
  freqs, times, spec = signal.spectrogram(audio,
                                          fs=sample_rate,
                                          window='hann',
                                          nperseg=nperseg,
                                          noverlap=noverlap,
                                          detrend=False)
  return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def normalize_spectrograms(batch, mean=None, std=None):
  if mean == None:
    mean = np.mean(batch, axis=0)
  if std == None:
    std = np.std(batch, axis=0)
  normalized_batch = (batch - mean) / std
  return mean, std, normalized_batch

def get_all_wav_files(dir_):
  # all_files = glob.glob(dir_ + 'train/audio/*/*.wav')
  all_files = glob.glob(dir_ + 'train/audio/no/*.wav')
  all_files += glob.glob(dir_ + 'train/audio/yes/*.wav')
  all_files += glob.glob(dir_ + 'train/audio/up/*.wav')
  all_files += glob.glob(dir_ + 'train/audio/down/*.wav')
  all_files += glob.glob(dir_ + 'train/audio/left/*.wav')
  all_files += glob.glob(dir_ + 'train/audio/right/*.wav')
  return all_files

def split_data(data, train_split=0.9):
  indices = np.random.choice(len(data), len(data))
  data = np.array(data)[indices]
  val = data[int(train_split * len(data)):]
  train = data[:int(train_split * len(data))]
  return train, val

def load_audio(files):

  data = []
  labels = []

  pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")

  for file_path in files:
    sample_rate, audio = load_wavfile(file_path)
    if len(audio) != 16000:
      continue
    freqs, times, spectrogram = log_spectrogram(audio, sample_rate)
    r = re.match(pattern, file_path)
    label = r.group(2)
    if label == '_background_noise_':
      label = 'silence'
    if label not in POSSIBLE_LABELS:
      label = 'unknown'

    data += [spectrogram]
    labels += [word2id[label]]

  return data, labels

def load_data(dir_):
  files = get_all_wav_files(dir_)
  train, val = split_data(files)
  del files
  train_X, train_y = load_audio(train)
  val_X, val_y = load_audio(val)

  train_X = np.stack(train_X)
  train_y = np.stack(train_y)

  val_X = np.stack(val_X)
  val_y = np.stack(val_y)

  return train_X, train_y, val_X, val_y, word2id, id2word

def get_mini_batch(X, y, bs=32):
  indices = np.random.choice(len(X), bs)
  return X[indices], y[indices]

# load_data('./data/')


# path = './data/train/audio/happy/fffcabd1_nohash_0.wav'
# sample_rate, audio = load_wavfile(path)

# freqs, times, spectrogram = log_spectrogram(audio, sample_rate)

# print(spectrogram.shape)
# print(times.shape)
# print(freqs.shape)
# print(times.min(), times.max())
# print(freqs.min(), freqs.max())
# print(np.unravel_index(np.argmax(spectrogram.T), spectrogram.T.shape))
# input()

# filename='tmp'

# fig = plt.figure(figsize=(14, 8))
# ax1 = fig.add_subplot(211)
# ax1.set_title('Raw wave of ' + filename)
# ax1.set_ylabel('Amplitude')
# ax1.plot(np.linspace(0, sample_rate/len(audio), sample_rate), audio)

# ax2 = fig.add_subplot(212)
# ax2.imshow(spectrogram.T, aspect='auto', origin='lower',
           # extent=[times.min(), times.max(), freqs.min(), freqs.max()])
# ax2.set_yticks(freqs[::16])
# ax2.set_xticks(times[::16])
# ax2.set_title('Spectrogram of ' + filename)
# ax2.set_ylabel('Freqs in Hz')
# ax2.set_xlabel('Seconds')

# plt.savefig('tmp.pdf', bbox_inches='tight', dpi=200)
