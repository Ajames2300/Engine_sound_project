# 2026.02.26
import os
import torch 
from torch.utils.data import Dataset
from scipy.io import wavfile

# Example usage:
# sample_rate, data = wavfile.read('data/raw/train_cut/engine1_good/pure_0.wav')
# Data is categorized into 4 classes: 
# engine1_good, engine2_broken, engine3_heavyload
# construct dataset according to train/test, train_cut/test_cut 

class EngineDataSet(Dataset):
    def __init__(self, dir: str):
        self.root_dir = dir
        self.sample = []
        self.class_idx = self._classes_to_index()
        self.dataset_construct()

    def __len__(self):
        return len(self.sample)
    
    def __getitem__(self, idx):
        data, label, sample_rate = self.sample[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.int16), sample_rate
    
    def _classes_to_index(self):
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        return {class_name: idx for idx, class_name in enumerate(classes)}

    def dataset_construct(self):
        for class_name, label in self.class_idx.items():
            class_dir  = os.path.join(self.root_dir, class_name)
            for wav_file_path in os.listdir(class_dir):
                
                # Avoid reading non-wav files
                if wav_file_path.endswith('.wav'):
                    
                    wav_path = os.path.join(class_dir, wav_file_path)
                    sample_rate, data = wavfile.read(wav_path)

                    # dataset is a list of tuples (data, label, sample_rate)
                    self.sample.append((data, label, sample_rate))
