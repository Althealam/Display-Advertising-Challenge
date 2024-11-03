from data_preprocess.data_preprocess import preProcess
from predict.gbdt_lr_predict import gbdt_lr_predict

if __name__ == '__main__':
    data = preProcess()
    continuous_feature = ['I'] * 13
    continuous_feature = [col + str(i + 1) for i, col in enumerate(continuous_feature)] 
    category_feature = ['C'] * 26
    category_feature = [col + str(i + 1) for i, col in enumerate(category_feature)] 
    gbdt_lr_predict(data, category_feature, continuous_feature)