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
from keras.callbacks import EarlyStopping, ModelCheckpoint

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


def data_for_ED_model(prime_train, prime_test):
    print('prime_train.shape:', prime_train.shape, 'prime_test.shape:', prime_test.shape)

    cols = prime_train.columns
    numeric_cols = ['Id', 'LotFrontage', 'LotArea', #  'MSSubClass',  'OverallQual', 'OverallCond',
                    'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                    'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',
                    'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                    'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',
                    'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                    'PoolArea', 'MiscVal', 'MoSold', 'YrSold'] # , 'SalePrice'
    y_cols = ['SalePrice']
    cat_cols = list(set(cols) - set(numeric_cols) - set(y_cols) )

    y = prime_train[y_cols]
    train = prime_train.drop(y_cols, axis=1)
    train_rows = prime_train.shape[0]

    merged = pd.concat([train, prime_train], axis=0, ignore_index=True)

    df_num = merged[numeric_cols]
    df_num = df_num.fillna(0)

    scaler = StandardScaler()
    df_num = scaler.fit_transform(df_num)
    df_num = pd.DataFrame(df_num, columns=numeric_cols) # use DataFrame because we care about column names.
    print('merged_num.shape:', df_num.shape)

    df_cat = merged[cat_cols]
    print('merged_cat.shape:', df_cat.shape)
    df_cat = df_cat.fillna('miss').applymap(str) # converts all to str, otherwise the numeric values went wrong.
    df_encoded_cat = pd.get_dummies(df_cat)
    print('merged_encoded_cat.shape:', df_encoded_cat.shape)

    df_transformed = pd.concat([df_num, df_encoded_cat], axis=1)
    print('merged_transformed.shape:', df_transformed.shape)

    x_merged = df_transformed

    x_test = df_transformed[train_rows:]
    print('x_merged.shape:', x_merged.shape, 'x_test.shape:', x_test.shape, 'y.shape:', y.shape, )
    return x_merged, x_test, y

# def data_for_ED_model(pred_cols, prime_train, prime_test):
#     train, _ = data_transformations(prime_train, pred_cols)
#     test, _ = data_transformations(prime_test, None)
#     x_merged = pd.concat([train, test], axis=0)
#
#     x_train, x_dev = train_test_split(x_merged, test_size=0.2, random_state=42)
#
#     x_test = test
#     # x_test.Depth = x_test.Depth.astype(np.bool).astype(np.float32)  # all other fields are also np.float32
#     # x_test.drop(['PIDN'], axis=1, inplace=True)
#     print('shapes: x_train, x_dev, x_test', x_train.shape, x_dev.shape, x_test.shape)
#     return x_train, x_dev, x_test

# def mcrmse(y_true, y_pred):
#     return MCRMSE(y_true.columns, y_true, y_pred)
#
# def MCRMSE(columns, y_true, y_pred):
#     return np.mean([mean_squared_error(y_true[col], y_pred[col]) for col in columns])

def create_ED_model(x_train):
    inp_shape = x_train.shape[1]

    dropout = 0.25
    inp = ks.Input(shape=(inp_shape,), dtype='float32')
    out = ks.layers.Dense(128, activation='relu')(inp)
    out = ks.layers.BatchNormalization()(out)
    out = ks.layers.Dropout(dropout)(out)
    out = ks.layers.Dense(64, activation='relu')(out)
    out = ks.layers.BatchNormalization()(out)
    out = ks.layers.Dropout(dropout)(out)
    out = ks.layers.Dense(128, activation='relu')(out)
    out = ks.layers.BatchNormalization()(out)
    out = ks.layers.Dropout(dropout)(out)
    out = ks.layers.Dense(inp_shape, activation='relu')(out)

    model = ks.Model(inp, out)
    model.compile(loss='mean_squared_error', optimizer=ks.optimizers.Adam(lr=3e-3))
    model.summary()
    return model

def train_ED_model(model, x_train):
    # Development: 0.12119085789122325
    # 0.08965547928152767 : 40 epochs
    batch_size = 32
    epochs = 200

    earlystopper = EarlyStopping(patience=int(epochs/10), verbose=1)
    checkpointer = ModelCheckpoint('models/EncoderDecoder.model', verbose=1, save_best_only=True)
    results = model.fit(x=x_train, y=x_train, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=2,
                        callbacks=[earlystopper, checkpointer])
    # for i in range(epochs):
    #     with timer('epoch {}'.format(i + 1)):
    #         model.fit(x=x_train, y=x_train, batch_size=batch_size, epochs=1, verbose=0)
    #         print(model.evaluate(x=x_dev, y=x_dev, batch_size=batch_size))
    print('results.history:', results.history)
    # score = model.evaluate(x=x_dev, y=x_dev, batch_size=batch_size)
    # print('score:', score)
    return model

def main(development=False, model_file='models/EncoderDecoder.model'):
    nrows = 10000 if development else None
    prime_train, prime_test = read_data(nrows)
    x_merged, x_test, y = data_for_ED_model(prime_train, prime_test)

    ed_model = create_ED_model(x_merged)
    ed_model = train_ED_model(ed_model, x_merged)
    ed_model.save(model_file)

if __name__ == "__main__":
    main(development=False)