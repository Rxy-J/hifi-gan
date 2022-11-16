import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize

from audio import melspectrogram, load_wav

def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                          for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [os.path.join(a.input_wavs_dir, x.split('|')[0] + '.wav')
                            for x in fi.read().split('\n') if len(x) > 0]
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, hparams, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.hparams = hparams
        self.segment_size = self.hparams.segment_size
        self.sampling_rate = self.hparams.sampling_rate
        self.split = split
        self.n_fft = self.hparams.n_fft
        self.num_mels = self.hparams.num_mels
        self.hop_size = self.hparams.hop_size
        self.win_size = self.hparams.win_size
        self.fmin = self.hparams.fmin
        self.fmax = self.hparams.fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio = load_wav(filename, 
                            self.sampling_rate, 
                            self.sampling_rate, 
                            self.win_size,
                            self.hop_size)
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if not self.fine_tuning:
            mel = melspectrogram(audio.squeeze(0).numpy(), self.hparams)
            mel = torch.from_numpy(mel)
        else:
            mel = np.load(os.path.join(self.base_mels_path, os.path.splitext(os.path.split(filename)[-1])[0] + '.npy'))
            mel = torch.from_numpy(mel)
        if len(mel.shape) < 3:
            mel = mel.unsqueeze(0)

        if self.split:
            frames_per_seg = math.ceil(self.segment_size / self.hop_size)

            if audio.size(1) >= self.segment_size:
                mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
            else:
                mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        return (mel.squeeze(), audio.squeeze(0), filename)

    def __len__(self):
        return len(self.audio_files)
