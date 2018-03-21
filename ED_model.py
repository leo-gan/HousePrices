# coding: utf-8

# ### v.02 : 
# - Latent Feature generation from different layers of ED (Encoder-Decoder) model.
# - Test data: Concatenate the original features and the latent features into new test datasets. 
# - Prediction model as lightGBM (or catboost). It outputs evaluation score + feature importance.
# - Get outputs from the Prediction model
# - Analyze, visualize outputs
# #### Small changes:
# - Choose the best ED model.
# - Optimize ED model:
#   - Try Dropout and BatchNormalization 

import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.model_selection import train_test_split, KFold
import keras as ks
from contextlib import contextmanager
import time
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from pandas_summary import DataFrameSummary

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print('[{}] done in {:.0f} s'.format(name, time.time() - t0))

def read_data(nrows):
    prime_train = pd.read_csv('input/train.csv', nrows=nrows)
    prime_test = pd.read_csv('input/test.csv', nrows=nrows)
    return prime_train, prime_test


def data_transformations(df):
    '''
    Categorical cols: Missed values replaced by '', transformed in one-hot.
    Numerical cols: Missed values replaced by 0, transformed with StandardScaler.
    :param df: Transformed DataFrame
    :param categ_cols: lost of categorical columns
    :param numeric_cols: list of numeric columns
    :return: df
    '''
    print('df.shape:', df.shape)

    cols = df.columns
    numeric_cols = ['Id', 'LotFrontage', 'LotArea', #  'MSSubClass',  'OverallQual', 'OverallCond',
                    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                    'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] # , 'SalePrice'

    y_cols = ['SalePrice']
    y = df[y_cols]

    df_num = df[numeric_cols]
    df_num = df_num.fillna(0)

    scaler = StandardScaler()
    df_num = scaler.fit_transform(df_num)
    df_num = pd.DataFrame(df_num, columns=numeric_cols)
    print('df_num.shape:', df_num.shape)

    cat_cols = list(set(cols) - set(numeric_cols) - set(y_cols))
    df_cat = df[cat_cols]
    print('df_cat.shape:', df_cat.shape)
    df_cat = df_cat.fillna('miss').applymap(str)
    df_encoded_cat = pd.get_dummies(df_cat)
    print('df_encoded_cat.shape:', df_encoded_cat.shape)

    df_transformed = pd.concat([df_num, df_encoded_cat], axis=1)
    print('df_transformed.shape:', df_transformed.shape)

    return df_transformed, y




def data_for_ED_model(pred_cols, prime_train, prime_test):
    train = prime_train.drop(pred_cols, axis=1)
    train = pd.concat([train, prime_test])
    train.Depth = train.Depth.astype(np.bool).astype(np.float32)  # all other fields are also np.float32
    train.drop(['PIDN'], axis=1, inplace=True)
    x_train, x_dev = train_test_split(train, test_size=0.2, random_state=42)
    print(x_train.shape, x_dev.shape)
    # In[8]:
    x_test = prime_test
    x_test.Depth = x_test.Depth.astype(np.bool).astype(np.float32)  # all other fields are also np.float32
    x_test.drop(['PIDN'], axis=1, inplace=True)
    print(x_test.shape)
    return x_train, x_dev, x_test

def mcrmse(y_true, y_pred):
    return MCRMSE(y_true.columns, y_true, y_pred)

def MCRMSE(columns, y_true, y_pred):
    return np.mean([mean_squared_error(y_true[col], y_pred[col]) for col in columns])

def create_ED_model(x_train):
    inp_shape = x_train.shape[1]

    inp = ks.Input(shape=(inp_shape,), dtype='float32')
    out = ks.layers.Dense(128, activation='relu')(inp)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.Dense(128, activation='relu')(out)
    out = ks.layers.Dense(inp_shape, activation='relu')(out)

    model = ks.Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    model.summary()
    return model

def train_ED_model(model, x_train, x_dev):
    # Development: 0.12119085789122325
    # 0.08965547928152767 : 40 epochs
    batch_size = 32
    epochs = 30
    for i in range(epochs):
        with timer('epoch {}'.format(i + 1)):
            model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=1, verbose=0)
            print(model.evaluate(x=x_dev, y=x_dev, batch_size=batch_size))

    return model.evaluate(x=x_dev, y=x_dev, batch_size=batch_size)

def main(development=False, model_file='models/EncoderDecoder.model'):
    # # Workflow
    # 1. Build the Encoder-Decoder model.
    # 2. Train it.
    # 3. Get the trained Encoder part of the model
    # 4. Use it to generate the additional features for the next model
    #
    # That means on 1. we don't need the predicted values at all. We can use both, train and test data to train the
    # Encoder-Decoder.

    # ## Prepare Data
    # We can use both, train and test data to train the Encoder-Decoder. There is no label data, becuse all input data acts as the label data.

    nrows = 10000 if development else None
    prime_train, prime_test = read_data(nrows)

    #print(prime_train.columns)
    # print(DataFrameSummary(prime_train).summary().T)
    # cols = list(prime_train.select_dtypes(include=[np.number]).columns.values)
    # print(len(cols), cols)
    # cat_cols = list(prime_train.select_dtypes(include=[np.object]).columns.values)
    # print(len(cat_cols), cat_cols)
    # cat_cols = list(prime_train.select_dtypes(include=[np.object]).columns.values)
    # print(len(cat_cols), cat_cols)
    df2, y = data_transformations(prime_train)
    # pred_cols = ['Ca', 'P', 'pH', 'SOC', 'Sand']  # excluding the 'PIDN' column
    #
    # x_train, x_dev, x_test = data_for_ED_model(pred_cols, prime_train, prime_test)
    # ed_model = create_ED_model(x_train)
    # score = train_ED_model(ed_model, x_train, x_dev)
    # ed_model.save(model_file)

main(development=True)