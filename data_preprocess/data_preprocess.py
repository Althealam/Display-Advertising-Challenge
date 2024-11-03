import pandas as pd

def preProcess():
    path = './data/'
    df_train = pd.read_csv(path + 'train.csv')
    df_test = pd.read_csv(path + 'test.csv')
    df_train.drop(['Id'], axis = 1, inplace = True)
    df_test.drop(['Id'], axis = 1, inplace = True)
    df_test['Label'] = -1
    data = pd.concat([df_train, df_test])
    data = data.fillna(-1)
    data.to_csv('./data/data.csv', index = False)
    return data
