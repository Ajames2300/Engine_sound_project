# Electric_motor_sound_project
Data source: Electrical motor Sound Data from Kaggle

1. 實作 Engine sound data 的特徵提取方法，並使用視覺化工具顯示特徵之分布情形。
2. 實作 Lasso 方法，使用 torch 模組，並嘗試改善 SGD 在 Lasso 方法中難以達到 L1 norm 的 regularization problem. Model weights 沒有達到稀疏性。

<img width="1000" height="400" alt="Lasso_model_weight" src="https://github.com/user-attachments/assets/f30053eb-71df-4ffc-8608-564bbd949083" />

  參考 sklearn 方法中 Lasso(linear_model.LogisticRegression) 中的方式使用 Soft threshold 以及 SAGA solver methods 改善 L1 norm 的 regularization.
  改善結果如下:
<img width="1000" height="400" alt="Lasso_model_weight_soft_threshold_saga" src="https://github.com/user-attachments/assets/c5a828bb-d248-4261-88bc-fc09acbf8981" />

  最終比較實作前後的Loss 收斂情形:
<img width="1000" height="400" alt="saga_sgd_comparison" src="https://github.com/user-attachments/assets/bdf8eeae-78ec-48aa-a777-2ce17c353207" />
