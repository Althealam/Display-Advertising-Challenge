import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

class LightGBM:
    def __init__(self, objective='binary', subsample=0.2, min_child_weight=0.5, colsample_bytree=0.7, 
                 num_leaves=64, learning_rate=0.05, n_estimators=100, random_state=2020):
        self.objective = objective
        self.subsample = subsample
        self.min_child_weight = min_child_weight
        self.colsample_bytree = colsample_bytree
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        y_pred = np.zeros(len(y))  # 初始预测值为0
        all_gbdt_feats = []  # 用于收集所有树的特征

        for i in range(self.n_estimators):
            # 随机采样部分数据和特征
            sample_indices = np.random.choice(len(X), int(self.subsample * len(X)), replace=False)
            feature_indices = np.random.choice(X.shape[1], int(self.colsample_bytree * X.shape[1]), replace=False)
        
            # 使用iloc确保获取行和列
            X_sample = X.iloc[sample_indices, feature_indices].values
            y_sample = y.iloc[sample_indices].values  # 假设y是一个pandas Series
        
            residual = y_sample - self._sigmoid(y_pred[sample_indices])
        
            # 创建并训练决策树
            tree = DecisionTreeRegressor(max_depth=int(np.log2(self.num_leaves)), random_state=self.random_state)
            tree.fit(X_sample, residual)

            # 记录每棵树的特征
            gbdt_feats = tree.predict(X.iloc[:, feature_indices].values).reshape(-1, 1)  # 确保是二维数组
            all_gbdt_feats.append(gbdt_feats)

            # 更新预测
            update = self.learning_rate * gbdt_feats.flatten()  # 一维数组
            y_pred += update

        # 在fit结束后，合并所有树的特征
        self.gbdt_feats_train = np.hstack(all_gbdt_feats)  # 合并成一个数组

    
    def predict_proba(self, X):
        y_pred = np.zeros(X.shape[0])
        
        for tree, feature_indices in self.trees:
            y_pred += self.learning_rate * tree.predict(X.iloc[:, feature_indices].values)
        
        return self._sigmoid(y_pred)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
