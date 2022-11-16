import numpy as np
from numpy import linalg as LA
import librosa
from scipy.io import wavfile
import soundfile as sf
import librosa.filters

def load_wav(wav_path, raw_sr, target_sr=16000, win_size=800, hop_size=200):
    audio = librosa.core.load(wav_path, sr=raw_sr)[0]
    if raw_sr != target_sr:
        audio = librosa.core.resample(audio,
                                      raw_sr,
                                      target_sr,
                                      res_type='kaiser_best')
        target_length = (audio.size // hop_size +
                         win_size // hop_size) * hop_size
        pad_len = (target_length - audio.size) // 2
        if audio.size % 2 == 0:
            audio = np.pad(audio, (pad_len, pad_len), mode='reflect')
        else:
            audio = np.pad(audio, (pad_len, pad_len + 1), mode='reflect')
    return audio


def save_wav(wav, path, sample_rate, norm=False):
    if norm:
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sample_rate, wav.astype(np.int16))
    else:
        sf.write(path, wav, sample_rate)


_mel_basis = None
_inv_mel_basis = None


def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sampling_rate // 2
    return librosa.filters.mel(hparams.sampling_rate,
                               hparams.n_fft,
                               n_mels=hparams.num_mels,
                               fmin=hparams.fmin,
                               fmax=hparams.fmax)


def _linear_to_mel(spectogram, hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectogram)


def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_size,
                        win_length=hparams.win_size)


def _amp_to_db(x, hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _normalize(S, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_db) /
                                                          (-hparams.min_db)) -
                           hparams.max_abs_value, -hparams.max_abs_value,
                           hparams.max_abs_value)
        else:
            return np.clip(
                hparams.max_abs_value * ((S - hparams.min_db) /
                                         (-hparams.min_db)), 0,
                hparams.max_abs_value)

    assert S.max() <= 0 and S.min() - hparams.min_db >= 0
    if hparams.symmetric_acoustic:
        return ((2 * hparams.max_abs_value) * ((S - hparams.min_db) /
                                               (-hparams.min_db)) -
                hparams.max_abs_value)
    else:
        return (hparams.max_abs_value * ((S - hparams.min_db) /
                                         (-hparams.min_db)))


def _denormalize(D, hparams):
    if hparams.allow_clipping_in_normalization:
        if hparams.symmetric_acoustic:
            return (
                ((np.clip(D, -hparams.max_abs_value, hparams.max_abs_value) +
                  hparams.max_abs_value) * -hparams.min_db /
                 (2 * hparams.max_abs_value)) + hparams.min_db)
        else:
            return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_db /
                     hparams.max_abs_value) + hparams.min_db)

    if hparams.symmetric_acoustic:
        return (((D + hparams.max_abs_value) * -hparams.min_db /
                 (2 * hparams.max_abs_value)) + hparams.min_db)
    else:
        return ((D * -hparams.min_db / hparams.max_abs_value) + hparams.min_db)


def _db_to_amp(x):
    return np.power(10.0, (x) * 0.05)


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    # Convert back to linear
    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db), hparams)

    return _griffin_lim(S**hparams.power, hparams)


def _griffin_lim(S, hparams):
    '''
    librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y


def _stft(y, hparams):
    return librosa.stft(y=y,
                        n_fft=hparams.n_fft,
                        hop_length=hparams.hop_size,
                        win_length=hparams.win_size)


def _istft(y, hparams):
    return librosa.istft(y,
                         hop_length=hparams.hop_size,
                         win_length=hparams.win_size)


def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D), hparams), hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S


def energy(wav, hparams):
    D = _stft(wav, hparams)
    magnitudes = np.abs(D).T  # [F, T]
    return LA.norm(magnitudes, axis=1)


def trim_silence(wav, hparams):
    '''
    Trim leading and trailing silence
    '''
    # These params are separate and tunable per dataset.
    unused_trimed, index = librosa.effects.trim(
        wav,
        top_db=hparams.preprocess.trim_top_db,
        frame_length=hparams.preprocess.trim_fft_size,
        hop_length=hparams.preprocess.trim_hop_size)

    num_sil_samples = \
        int(hparams.preprocess.num_silent_frames * hparams.data.hop_size)
    start_idx = max(index[0] - num_sil_samples, 0)
    stop_idx = min(index[1] + num_sil_samples, len(wav))
    trimmed = wav[start_idx:stop_idx]

    return trimmed
