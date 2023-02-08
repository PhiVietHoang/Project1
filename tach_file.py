import pandas as pd
import numpy as np

def train_validate_test_split(df, train_percent=.7, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

# dfData = pd.read_csv('C:/Users/ADMIN/OneDrive/Documents/GR1/data set/Algerian_forest_fires_dataset_UPDATE.csv')
# listColumns = dfData.columns.to_list()
# train = pd.DataFrame()
# validate = pd.DataFrame()
# test = pd.DataFrame()
# columnsLabelName = listColumns[len(listColumns) - 1]
# listLabel = dfData[columnsLabelName].unique().tolist()

# for y in listLabel:
#     dfy = dfData[dfData[columnsLabelName]==y]
#     train1, validate1, test1 = train_validate_test_split(dfy)
#     train = pd.concat([train,train1])
#     validate = pd.concat([validate,validate1])
#     test = pd.concat([test,test1])

# train.to_csv(f'C:/Users/ADMIN/OneDrive/Documents/GR1/Data_da_tach_2/train_Algerian_forest_fires_dataset_UPDATE.csv', index = False)
# validate.to_csv(f'C:/Users/ADMIN/OneDrive/Documents/GR1/Data_da_tach_2/validate_Algerian_forest_fires_dataset_UPDATE.csv', index = False)
# test.to_csv(f'C:/Users/ADMIN/OneDrive/Documents/GR1/Data_da_tach_2/test_Algerian_forest_fires_dataset_UPDATE.csv', index = False)

# print(listLabel)
# print(train)
# print(test)
# print(validate)
