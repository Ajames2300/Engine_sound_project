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
from sklearn import linear_model
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
    

    # To arr, to tensor 
    entire_static_feature = torch.tensor([[d["mean"], d["variance"], d["std"], d["rms"]] for d in entire_static_feature])
    entire_frequency_feature = torch.tensor(np.array(peaks_height))
    

    X_train = torch.concat((entire_static_feature, entire_frequency_feature), dim = 1)
    y_train = torch.tensor(label, dtype=torch.int64)

    del entire_static_feature, entire_frequency_feature

    # Normalize validation data with training data information 
    scalar = StandardScaler()
    scalar.fit(X_train)

    X_train = torch.tensor(scalar.transform(X_train), dtype= torch.float32)
    

    # =========== Lasso training ========== # 
    model = LassoModel(dim_input = X_train.shape[1], num_class = 3)

    lr = 0.001 # 實作 saga, initial learning rate would be range from 0.001 to 0.01 
    criteria =  nn.CrossEntropyLoss()  # Classification tasks
    lambda_l1 = 1  # Equals to C = 0.5 in sklearn Lasso 
    batch_size = 16 
    epoch = 200 
    tol = 0.001
    min_epoch = 20
   
    # If batch size is 1, then each sample is a batch, and the batch indices should be [0, 1, 2, ..., N-1]
    if batch_size == 1:       
        batch_indices = np.concatenate([np.arange(batch_size - 1, len(X_train) - 1, batch_size), [len(X_train) - 1]]) 
    else:
        batch_indices = np.concatenate([
        [0], 
        np.arange(batch_size - 1, len(X_train) - 1, batch_size), 
        [len(X_train) - 1]])

    # Calculate iteration numbers and batch indices 
    iter_num = len(batch_indices) - 1

    # Saga solver with soft thresholding 
    # Div：N * D * num_class
    N, D = X_train.shape
    grad_table = torch.zeros(N, D, 3)
    avg_grad = torch.zeros(D, 3)

    loss_save = []

    for i in range(epoch):
        # for idx in range(iter_num):
        for idx in range(iter_num):

            model.train()
            # optimizer.zero_grad()

            # Batch 
            xtrain = X_train[batch_indices[idx]: batch_indices[idx+1]]
            ytrain = y_train[batch_indices[idx]: batch_indices[idx+1]]

            # forward
            output = model(xtrain)
            loss = criteria(output, ytrain)

            # Calculate the gradient part 
            loss.backward() 

            # Soft threshold
            threshold = lr * lambda_l1

            # 取得當前權重的梯度 (假設只優化 weight，不優化 bias)
            current_grad = model.linear.weight.grad.data.clone().T # 轉置以匹配 (D, C)
            
            if lambda_l1 > 0:
                for name, p in model.named_parameters():
                    # Update weight only, bias not
                    if 'bias' not in name:

                        # 取得上個 epoch 中該 batch 的 grad
                        old_grad = grad_table[idx]
     
                        # SAGA : [新梯度 - 舊梯度 + 平均梯度]
                        saga_update_dir = current_grad - old_grad + avg_grad

                        # Weight update 
                        p.data -= lr * saga_update_dir.T

                        # sign function to avoid value near zero, then optimize w(k+1) with minus of half of threshold, 
                        # learning rate is the steps size provided with tau value
                        p.data = torch.sign(p.data) * torch.clamp((p.data.abs() - threshold), min=0)

                        # update grad_table and avg_grad
                        grad_table[idx] = current_grad
                        avg_grad += (current_grad - old_grad) / N
                        
                    pass

        # Append each epoch
        loss_save.append(loss.item()) 

        if (i + 1)% 5 == 0:
            print(f'Epoch [{i+1}/{epoch}], Loss: {loss.item():.4f}')  
        
        # Convergence check
        if i > min_epoch and loss.item() < tol:
            print(f'Convergence reached at epoch {i+1}, Loss: {loss.item():.4f}')
            break

        # Prevent overfit with early stopping
        if i > min_epoch and all(loss.item() > prev_loss for prev_loss in loss_save[-6:-1]):
            print(f'Early stopping at epoch {i+1}, Loss: {loss.item():.4f} with previous losses')
            break

    # Save trained data nomalization info 
    with open('src/utils/scalar.pkl', 'wb') as f:
        pickle.dump(scalar, f)

    # Save trained model 
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
    
    # Save loss history for visualization
    save_loss_name = "src/utils/lasso_applied_batch_loss.npy"
    np.save(save_loss_name, np.array(loss_save))

    del model, criteria, grad_table, avg_grad, X_train, y_train

    pass    
    


