資料source: Engine Sound Motor from Kaggle
1. 實作 Engine Sound Data 的特徵提取方法，並使用視覺化工具顯示特徵之分布情形(Visualization)
2. 實作 Lasso 方法，使用 torch 模組，並嘗試改善 SGD 在 Lasso 方法中難以達到 L1 norm 的 regularization problem. Model weights 沒有達到稀疏性
<img width="1000" height="400" alt="Lasso_model_weight" src="https://github.com/user-attachments/assets/b9814739-1e7f-4427-bd27-d1b0fb8383bb" />
參考 sklearn 方法中 Lasso(linear_model.LogisticRegression) 中的方式使用 Soft threshold 改善 L1 norm 的 regularization.
搭配 SAGA solver 使用小批次(mini-batch) 修正 Lasso 在使用 SGD 方法無法在0附近計算梯度，與 Lasso 模型特性相悖。
<img width="1000" height="400" alt="Lasso_model_weight_soft_threshold_saga" src="https://github.com/user-attachments/assets/aba0ed3c-0598-4c98-8834-c3bf2d5fc3f5" />
