import os 
import sys 
import torch 
import torch.nn as nn  
import torch.optim as optim 
import numpy as np

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pickle

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# Train model using final set of hyperparameters 
if __name__ == "__main__":
    from features.feature import FreqFeatureExtractor, StaticFeatureExtractor
    from data.data_loader import EngineDataSet
    from models.model import LassoModel

    # ---------------------------------------------------- #
    TrainEngineData = EngineDataSet('data/raw/train_cut/')
    label = TrainEngineData.label
    sample = TrainEngineData.sample

    # Static
    static_feature = StaticFeatureExtractor(data=sample)
    entire_static_feature = static_feature.make_static_features()

    # Frequency
    freq_feature = FreqFeatureExtractor(data = sample)
    
    _, psd_pxx = freq_feature.psd(sample_rate= TrainEngineData.sample_rate[0], 
                                  nfft = 2048, 
                                  window = "hann", 
                                  scaling = "density")
    
    _, peaks_height = freq_feature.top_peaks_finding(psd_feature = psd_pxx,
                                                     height = 0)
    

    # TODO: 1. Train Lasso with pytorch version and with static features
    #       2. Train Lasso with sklearn version and with static features
    #       3. Train CNN with pytorch version and with features as STFT

    # To arr, to tensor 
    entire_static_feature = torch.tensor([[d["mean"], d["variance"], d["std"], d["rms"]] for d in entire_static_feature])
    entire_frequency_feature = torch.tensor(np.array(peaks_height))
    

    X_train = torch.concat((entire_static_feature, entire_frequency_feature), dim = 1)
    y_train = torch.tensor(label, dtype=torch.int64)

    del entire_static_feature, entire_frequency_feature

    # Normalize validation data with training data information 
    scalar = StandardScaler()
    scalar.fit(X_train)

    X_train = torch.tensor(scalar.fit_transform(X_train), dtype= torch.float32)
    

    # =========== Lasso training ========== # 
    model = LassoModel(dim_input = X_train.shape[1], num_class = 3)
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    criteria =  nn.CrossEntropyLoss()  # Classification tasks
    lambda_l1 = 1
    epoch = 200
    min_epoch = 20 
    tol = 1e-3
    loss_save = []

    for i in range(epoch):

        optimizer.zero_grad()

        # forward
        output = model(X_train)
        loss = criteria(output, y_train)

        # L1 penalty (exclude bias parameters)
        l1_loss = sum(p.abs().sum() for name, p in model.named_parameters() if not name.endswith('bias'))

        # Calculate entire loss 
        total_loss = loss + lambda_l1*l1_loss

        # Update weights
        total_loss.backward()
        optimizer.step()

        if (i + 1)% 5 == 0:
            print(f'Epoch [{i+1}/{epoch}], Loss: {total_loss.item():.4f}')

        loss_save.append(total_loss.item())

        # Convergence check
        if i > min_epoch and total_loss.item() < tol:
            print(f'Convergence reached at epoch {i+1}, Loss: {loss.item():.4f}')
            break

        # Prevent overfit with early stopping
        if i > min_epoch and all(total_loss.item() > prev_loss for prev_loss in loss_save[-6:-1]):
            print(f'Early stopping at epoch {i+1}, Loss: {loss.item():.4f} with previous losses')
            break  

    # Save trained model and trained data nomalization info 
    with open('src/utils/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)

    saved_model_name = "src/utils/lasso_model.pt"
    torch.save(model.state_dict(), saved_model_name)
    
    weights = model.linear.weight.detach().numpy()
    feature_name = ['Mean', 'Variance', 'STD', 'RMS', 'Peak_1', 'Peak_2', 'Peak_3']
    plt.figure(figsize=(10, 4))
    sns.heatmap(weights, annot=True, cmap='RdBu', center=0,
                yticklabels=['Class 0', 'Class 1', 'Class 2'],
                xticklabels=[name for name in feature_name])
    plt.title("Feature Importance Heatmap")
    plt.show()

    save_loss_name = "src/utils/lasso_loss.npy"
    np.save(save_loss_name, np.array(loss_save))
    pass    
    

    

