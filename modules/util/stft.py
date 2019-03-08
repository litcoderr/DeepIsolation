# -*- coding: utf-8 -*-

#%%
import librosa
import librosa.display
import numpy as np
import scipy.io.wavfile as wv
import os

class STFTConverter:
    def __init__(self):
        pass
    
    @staticmethod
    def get_stft_from_wave(wave,comp):
        D = librosa.core.stft(wave, n_fft=4096)  # 1+2046/2 =1,024
        if comp == False:
            mag, phase = librosa.magphase(D)
            db, angle = librosa.amplitude_to_db(mag), np.angle(phase)
            db, angle = np.transpose(db, (1, 0)), np.transpose(angle, (1, 0))
        elif comp == True:
            db, angle = np.transpose(D.real,(1,0)), np.transpose(D.imag,(1,0))
        return db, angle
    
    def get_stft(self, wave, comp=False):
        """
        Returns [log_magnitude, radian_phase], cropped_wave
        """
        db, angle = self.get_stft_from_wave(wave,comp)
        converted = np.stack([db, angle], axis=0)
        return converted

    def stft2wav(self, stft):
        db, angle = np.transpose(stft[0], (1, 0)), np.transpose(stft[1], (1, 0))
        mag = librosa.core.db_to_amplitude(db)
        phase = np.cos(angle) + 1j * np.sin(angle)
        D = mag * phase
        y_hat = librosa.core.istft(D)
        return y_hat
