import os 
import sys
from scipy.fft import fft
import torch
import numpy as np
import scipy.signal as signal  # STFT & FFT choice
from statsmodels.tsa.stattools import acf  # Autocorrelation function   

class FreqFeatureExtractor:
    def __init__(self, data):
        self.data = data
        self.fft_result = []
        self.peaks = []
        self.spectrogram_result = []
    
    def psd(self, **kwargs):
        psd = []
        for seg in range(len(self.data)):
            f, pxx = signal.periodogram(self.data[seg][0], fs=self.data[seg][2], **kwargs)  # Calculate power spectral density using periodogram
            psd.append((f, pxx))
        
        return psd 

    def fft_n(self, output_len: int|None = None, **kwargs):
        # Compute FFT of self.data with given sample_rate
        fft_features = []
        
        # Caculate the amplitude of fft result, and find peaks in the FFT result
        for seg in range(len(self.data)):
            # z-score normalization
            normalized_data = (self.data[seg][0] - self.data[seg][0].mean()) / (self.data[seg][0].std() + 1e-8)
            fft_features.append(abs(np.fft.fft(normalized_data, n = output_len, **kwargs))/len(normalized_data))  # Normalize by the length of the signal to get the amplitude
        
        self.fft_result = fft_features

    def peaks_finding(self, feature, *args, **kwargs):
        peaks = []
        
        for seg in range(len(feature)):
            idx, prop = signal.find_peaks(feature[seg][0], *args, **kwargs)  # Find peaks in the data with a minimum height threshold
            
            if len(idx > 0):
                peak_values = [(idx[i], prop['peak_heights'][i]) for i in range(len(idx))]
                peak_values.sort(key=lambda x: x[1], reverse=True)  # 按值降序排列
                peaks.append((idx, prop))

            else:
                peaks.append(([], {}))  

            # peaks.append(signal.find_peaks(feature[seg][0], *args, **kwargs))  # Find peaks in the data with a minimum height threshold
        
        return peaks
        
class StaticFeatureExtractor:
    def __init__(self, data):
        self.data = data
        
    def autocorrelation(self, lags: int|None = None):
        acf_feature = []
        for seg in range(len(self.data)):
            # acf function from statsmodels can calculate autocorrelation with FFT method and conduct normalization.
            acf_feature.append(acf(self.data[seg][0], nlags=lags, fft=True))

        return acf_feature

    def mean(self):
        return [seg[0].mean() for seg in self.data]
    
    def rms(self):
        return [torch.sqrt(torch.mean(torch.square(torch.tensor(seg[0])))) for seg in self.data]
    
    def std(self):
        return [np.std(seg[0]) for seg in self.data]
    
    def make_static_features(self):
        entire_feature = []
        for seg in range(len(self.data)):
            entire_feature.append({
                "mean": self.data[seg][0].mean(),
                "std": np.std(self.data[seg][0]),
                "rms": np.sqrt(np.mean(np.square(np.array(self.data[seg][0]))))
                })
            
        return entire_feature
            
        

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    #------------------------------------------------------#
    from data.data_loader import EngineDataSet
    from matplotlib import pyplot as plt
    #------------------------------------------------------#

    raw_data = EngineDataSet('data/raw/train_cut')

    # feature = StaticFeatureExtractor(raw_data)
    # sample_rate = dataset[0][2]
    
    # entire_feature = feature.make_static_features()
    
    
    pass
    

    # === frequency domain features === # 
    feature = FreqFeatureExtractor(raw_data)

    psd = feature.psd(nfft = 2048, window = "hann", scaling = "density")  
    
    plt.plot(psd[-2][0], psd[-2][1])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density")   
    plt.show()





    # ------------------------------------------------------#
    # static_features = StaticFeatureExtractor(dataset)
    # acf_feature = static_features.autocorrelation(lags=len(dataset[0][0])//2)  # Calculate autocorrelation with half the length of the signal as lags
    
    pass
    # n_fft_features, peaks = feature.fft_n(output_len=int(sample_rate/2))  # 1 second window for FFT

    # plt.plot(n_fft_features[0])
    # plt.plot(peaks[0][0], n_fft_features[0][peaks[0][0]], "x")
    # plt.plot(np.zeros_like(n_fft_features[0]), "--", color="gray")
    # plt.show()

    pass
    # 繪圖
    # from statsmodels.graphics.tsaplots import plot_acf
    # plot_acf(acf_feature[-1], lags=1000)
    # plt.show()


    # # Comparison
    # fig, ax = plt.subplots(2,1, figsize=(10, 6))
    
    # fs = dataset[0][2]
    # N = len(dataset[0][0])

    # freq = np.fft.fftfreq(len(fft_features), d=1/fs)
    # ax[0].plot(freq[:len(freq)//2], fft_features[0][:len(fft_features)//2], label='FFT')
    # ax[1].plot(freq[:len(freq)//2], n_fft_features[0][:len(n_fft_features)//2], label='Normalized FFT')
    # ax[0].legend()
    # ax[1].legend()

    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.show()

