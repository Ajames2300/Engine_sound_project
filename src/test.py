import os 
import sys 
import numpy as np 
import torch 
import torch.nn as nn  
import torch.optim as optim 
import numpy as np 
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)


if __name__ == "__main__":
    from features.feature import FreqFeatureExtractor, StaticFeatureExtractor
    from data.data_loader import EngineDataSet
    from models.model import LassoModel

    # ---------------------------------------------------- #
    TestEngineData = EngineDataSet('data/raw/test_cut/')
    test_label = TestEngineData.label
    test_sample = TestEngineData.sample

    test_static_feature = StaticFeatureExtractor(data = test_sample)
    entire_static_feature = test_static_feature.make_static_features()

    # Frequency feature(log band power ratio)
    freq_feature = FreqFeatureExtractor(data = test_sample)
    log_band_power_ratios = freq_feature.make_freq_features(feature_type="log_power_ratios")
    
    # To arr, to tensor 
    entire_static_feature = torch.tensor([[d["mean"], d["variance"], d["std"], d["rms"]] for d in entire_static_feature])
    entire_frequency_feature = torch.tensor(np.array(log_band_power_ratios))
    entire_feature = torch.concat((entire_static_feature, entire_frequency_feature), dim = 1)
    
    del entire_static_feature, entire_frequency_feature

    with open('src/utils/scalar.pkl', 'rb') as f:
        trained_scalar = pickle.load(f)

    X_Test = torch.tensor(trained_scalar.transform(entire_feature), dtype= torch.float32)
    y_Test = torch.tensor(test_label, dtype=torch.int64)

    # TODO: Load model 
    model_pth = "src/utils/lasso_model.pt"
    trained_model = LassoModel(dim_input = X_Test.shape[1], num_class = 3)
    trained_model.load_state_dict(torch.load(model_pth)) # Load model dict
    
    trained_model.eval()
    with torch.no_grad():
            predict_result = torch.softmax(trained_model(X_Test), dim = 1)
            result = torch.max(predict_result, dim = 1)
    
    acc = sum(np.array(test_label) == result.indices)/len(test_label)

    print(f'Test result score, Acc: {acc:.4f}')
    pass
