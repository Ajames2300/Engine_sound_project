import os 
import sys
from scipy.fft import fft
from scipy.stats import kurtosis
import torch
import numpy as np
import scipy.signal as signal  # STFT & FFT choice
from statsmodels.tsa.stattools import acf  # Autocorrelation function   

class FreqFeatureExtractor:
    def __init__(self, data):
        self.data = data
        self.fft_result = []
        self.flag  = False # Flag to check if frequency features have been extracted
    
    def make_freq_features(self, feature_type: str):
        feature = None
        if feature_type not in ["psd_peaks", "log_power_ratios"]:
            raise ValueError("Invalid feature type. Supported types are 'psd_peaks' and 'log_power_ratios'.")
        
        if feature_type == "psd_peaks":
            _, psd_pxx = self.psd(sample_rate=44100, nfft=2048, window="hann", scaling="density")
            _, peaks_height = self.top_peaks_finding(psd_feature=psd_pxx, height=0)
            feature = peaks_height

        elif feature_type == "log_power_ratios":
            self.fft_n(output_len = 2048, window = "hann")
            feature = self.log_band_power_ratio()



        return feature

    def psd(self, sample_rate, **kwargs):
        psd_f = []
        psd_pxx = []
        for seg in range(len(self.data)):
            f, pxx = signal.periodogram(self.data[seg], fs=sample_rate, **kwargs)  # Calculate power spectral density using periodogram
            psd_f.append(f)
            psd_pxx.append(pxx)
        
        self.flag = True 
        
        return psd_f, psd_pxx

    def fft_n(self, output_len: int|None = None, **kwargs):
        # Compute FFT of self.data with given sample_rate
        fft_features = []
        
        # Caculate the amplitude of fft result, and find peaks in the FFT result
        for seg in range(len(self.data)):
            # z-score normalization
            normalized_data = (self.data[seg] - self.data[seg].mean()) / (self.data[seg].std() + 1e-8)
            
            # Apply window if specified
            if 'window' in kwargs:
                window = signal.get_window(kwargs['window'], len(normalized_data))
                normalized_data = normalized_data * window
                kwargs = {k: v for k, v in kwargs.items() if k != 'window'}
            
            fft_features.append(abs(np.fft.fft(normalized_data, n = output_len, **kwargs))/len(normalized_data))  # Normalize by the length of the signal to get the amplitude

        self.fft_result = fft_features
        self.flag = True


    def top_peaks_finding(self, psd_feature, *args, **kwargs):

        # Find top 3 peaks in the feature with a minimum height threshold
        peaks_indices = []
        peaks_height = []
        
        for seg in range(len(self.data)):
            idx, prop = signal.find_peaks(x = psd_feature[seg], *args, **kwargs)  # Find peaks in the data with a minimum height threshold
            
            if len(idx > 0):
                peak_values = [(idx[i], prop['peak_heights'][i]) for i in range(len(idx))]
                peak_values.sort(key=lambda x: x[1], reverse=True)  # Sort peaks by height in descending order
                
                top_3_peaks = peak_values[:3]
                peaks_indices.append([p[0] for p in top_3_peaks])  # Top 3 peaks indices
                peaks_height.append([p[1] for p in top_3_peaks])

            else:
                peaks_indices.append([])
                peaks_height.append({})
        
        return peaks_indices, peaks_height
    
    def stft(self, sample_rate, **kwargs):
        stft_result = []
        for seg in range(len(self.data)):
            f, t, zxx = signal.stft(self.data[seg], fs=sample_rate, **kwargs)  # Compute the Short-Time Fourier Transform (STFT) of the signal
            stft_result.append({"f": f, "t": t, "zxx": zxx})
        
        self.flag = True

        return stft_result
    
    def log_band_power_ratio(self, band_ranges=None):

        if self.flag == False:
            raise ValueError("Please calculate the FFT features before calculating the band power ratio.")  
        
        sample_rate = 44100 
        n_fft = self.fft_result[0].shape[0] 
        sample_rate_steps  = sample_rate / n_fft /2  # Frequency resolution of the FFT

        
        band_ranges = band_ranges if band_ranges is not None else [(0.5, 3000), (3000, 8000), (8000, 13000), (13000, 18000), (18000, sample_rate/2)]
        band_ranges_idx = [(int(low / sample_rate_steps), int(high / sample_rate_steps)) for low, high in band_ranges]
        
        log_band_power_ratios = []
        for seg in range(len(self.fft_result)):
            band_powers = []
            for low, high in band_ranges:
                band_power_range = np.abs(self.fft_result[seg][band_ranges_idx[band_ranges.index((low, high))][0]:band_ranges_idx[band_ranges.index((low, high))][1]])
                band_power_square_sum = np.sum(band_power_range ** 2)
                band_powers.append(np.log(band_power_square_sum + 1e-8))  # Add a small constant to avoid log(0))

            total_power = np.sum(band_powers)

            log_band_power_ratios.append([band_power / total_power for band_power in band_powers])

        return log_band_power_ratios

class StaticFeatureExtractor():
    def __init__(self, data):
        self.data = data
        self.flag = False

    def __getitem__(self, key):
        if not self.flag: 
            return self 
        else:
            return False
    
        
    def autocorrelation(self, lags: int|None = None):
        acf_feature = []
        for seg in range(len(self.data)):
            # acf function from statsmodels can calculate autocorrelation with FFT method and conduct normalization.
            acf_feature.append(acf(self.data[seg], nlags=lags, fft=True))

        return acf_feature

    def mean(self):
        return [seg.mean() for seg in self.data]
    
    def rms(self):
        return [torch.sqrt(torch.clamp(torch.mean(torch.square(torch.tensor(seg))))) for seg in self.data]
    
    def std(self):
        return [np.std(seg) for seg in self.data]
    
    def make_static_features(self):
        entire_feature = []
        for seg in range(len(self.data)):
            entire_feature.append({
                "mean": self.data[seg].mean(),
                "variance": self.data[seg].var(),
                "std": np.std(self.data[seg]),
                "rms": np.sqrt(np.mean(np.square(self.data[seg]))),
                "kurt": kurtosis(self.data[seg])
                })
            pass
            # if np.isnan(entire_feature[seg]["rms"]):
            #     pass
        
        self.flag = True

        return entire_feature
            
        

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    #------------------------------------------------------#
    from data.data_loader import EngineDataSet
    from matplotlib import pyplot as plt
    #------------------------------------------------------#

    # raw_data = EngineDataSet('data/raw/train_cut')

    # ---------------------------------------------------- #
    EngineData = EngineDataSet('data/raw/train_cut/')
   
    # Construct static datasets 
    static_features_good = StaticFeatureExtractor(EngineData.sample)
    f = static_features_good.make_static_features()

    pass
    
    # === frequency domain features === # 
    feature = FreqFeatureExtractor(EngineData.sample)
    # psd_f, psd_pxx = feature.psd(sample_rate= EngineData.sample_rate[0], 
    #                              nfft = 2048, 
    #                              window = "hann", 
    #                              scaling = "density")
    
    # peaks_indices, peaks_height = feature.top_peaks_finding(psd_feature = psd_pxx,
    #                                                         height = 0)

    # plt.plot(psd_f[-2], psd_pxx[-2]) # Heavyload engine
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power Spectral Density")   
    # plt.show()
    
    feature.fft_n(output_len = 2048, window = "hann")
    log_band_power_ratios = feature.log_band_power_ratio()

    pass 

    # stft time resolution is set as 0.2s, and the overlap is half of the window length
    # stft_result = feature.stft(nperseg= 0.2*raw_data.sample_rate[0], window="hann", boundary=None)  # Compute the Short-Time Fourier Transform (STFT) of the signal with specified parameters
    
    # seg = 150 # Example segment index for visualization
    # plt.pcolormesh(stft_result[seg]['t'], stft_result[seg]['f'], np.abs(stft_result[seg]['zxx']), shading='gouraud')
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.title('STFT Magnitude')
    # plt.colorbar(label='Magnitude')     
    # plt.show()

    pass
