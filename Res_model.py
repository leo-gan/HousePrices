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

# In[1]:


import pandas as pd
import numpy as np
from sklearn import svm, cross_validation
from sklearn.model_selection import train_test_split, KFold
import keras as ks
from contextlib import contextmanager
import time
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from keras import backend as K
import json

import ED_model

from pandas_summary import DataFrameSummary

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

# =================================================================================
# # Latent Feature generation from different layers of ED (Encoder-Decoder) model
# 1. Load ED model
# 2. Remove the last FC layer
# 3. Predict the latent feature from model
# 4. Go to 2 till ED has only the top layer

def get_layer_output(layer_num, layer_outs):
    assert layer_num >= 0 and layer_num < len(layer_outs)
    layer_out = layer_outs[layer_num][0]
    cols = ['ly'+str(layer_num)+'_'+str(col) for col in range(layer_out.shape[1])]
    return pd.DataFrame(data=layer_out, columns=cols)

def get_ED_outputs(model_name, x_train):
    ed_model = ks.models.load_model(model_name)
    # ed_model_layers_num = len(ed_model.layers) - 1
    # print('ED_model number of layers (-1):', ed_model_layers_num)  # Input layer does not count
    print(ed_model.summary())

    ed_model.layers.pop() # not interested in the last layer
    print(ed_model.summary())

    features = ed_model.predict(x_train)
    print('features.shape:',  features.shape)

    inp = ed_model.input  # input placeholder
    outputs = [layer.output for layer in ed_model.layers]  # all layer outputs
    functors = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]
    layer_outs = [func([x_train, 1.]) for func in functors]
    return [get_layer_output(l, layer_outs) for l in range(1, len(layer_outs))]

# def data_for_res_model(prime_train, pred_cols):
#     y_train = prime_train[pred_cols]
#     x_train = prime_train.drop(pred_cols, axis=1)
#
#     x_train.Depth = x_train.Depth.astype(np.bool).astype(np.float32)  # all other fields are also np.float32
#     x_train.drop(['PIDN'], axis=1, inplace=True)
#     print(x_train.shape, y_train.shape)
#     return x_train, y_train

def concat_data(case, layer_outs_dfs, df):
    if case == None: return df
    dfs = [layer_outs_dfs[c] for c in case]
    return pd.concat(dfs + [df], axis=1)

def predict(x_train, x_dev, y_train, y_dev):
    predictions = pd.DataFrame()
    feature_importances = {}
    scores = {}
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': {'l2', 'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0
    }
    for col in y_train.columns:
        print(col, ': Training...')
        lgb_train = lgb.Dataset(x_train.values, y_train[col].values)
        lgb_eval = lgb.Dataset(x_dev.values, y_dev[col].values, reference=lgb_train)

        res_model = lgb.train(params, lgb_train,
                                num_boost_round=20,
                                valid_sets=lgb_eval,
                                early_stopping_rounds=5
                              )

        res_model.save_model('models/lightgbm_'+col+'.model')
        print('       Predicting...')
        predictions[col] = res_model.predict(x_dev.values, num_iteration=res_model.best_iteration)
        scores[col] = mean_squared_error(y_dev[col].values, predictions[col])
        print('       rmse =', scores[col])
        feature_importances[col] = res_model.feature_importance()
    return predictions, scores, feature_importances

def calc_scores(layer_outs_dfs, x, y, verbose=True):
    scores_all = []
    cases = [None, [2], [5], [8], [2, 5], [5, 8], [2, 8], [2, 5, 8]]
    for case in cases:
        x_conc = concat_data(case, layer_outs_dfs, x)
        if verbose: print('   x_conc.shape, x.shape:', x_conc.shape, x.shape)

        x_train, x_dev, y_train, y_dev = train_test_split(x_conc, y, test_size=0.2, random_state=42)
        if verbose: print('   x_train.shape, x_dev.shape, y_train.shape, y_dev.shape:', x_train.shape, x_dev.shape, y_train.shape,
              y_dev.shape)

        predictions, scores, feature_importances = predict(x_train, x_dev, y_train, y_dev)
        if verbose: print('   case: scores:', case, scores)  # , feature_importances)
        scores_all.append((case, scores))
    if verbose: print(scores_all)
    return [(score[0], np.mean(list(score[1].values()))) for score in scores_all], scores_all, feature_importances

def write_data(file_name, vars):
    with open('models/'+file_name+'.json', 'w') as outfile:
        json.dump(vars, outfile)

# def read_data(file_name):
#     with open(file_name, 'r') as data_file:
#         return json.load(data_file)

def main(res_model_name, development=False, ED_model_file='models/EncoderDecoder.model'):
    nrows = 10000 if development else None
    prime_train, prime_test = ED_model.read_data(nrows)
    x_merged, x_test, y_train = ED_model.data_for_ED_model(prime_train, prime_test)
    x_train = x_merged[:x_test.shape[0]]

    layer_outs_dfs = get_ED_outputs(ED_model_file, x_train)
    _ = [print(ly.shape) for ly in layer_outs_dfs]
    #x, y = data_for_res_model(prime_train, pred_cols)
    scores_avg, scores_all, feature_importances = calc_scores(layer_outs_dfs, x_train, y_train)

    print(scores_avg, scores_all)
    write_data(res_model_name+'_scores', [scores_avg, scores_all])
    write_data(res_model_name+'_feature_importances', feature_importances.tolist())


#main('AfricaSoil_01')
# tmp = [ly[0] for ly in layer_outs]
# layer_outs_np = np.vstack((ly[0] for ly in layer_outs))
# print(layer_outs_np.shape)
# layer_outs_np_col_num = [ly[0].shape[1] for ly in layer_outs]
# print(layer_outs_np_col_num)
# layer_outs_np_col_names = ['ly'+str(l)+'_'+str(col)
#                            for l, col_num in enumerate(layer_outs_np_col_num)
#                            for col in range(col_num)
#                            ]
# sample['Ca'] = preds[:, 0]
# sample['P'] = preds[:, 1]
# sample['pH'] = preds[:, 2]
# sample['SOC'] = preds[:, 3]
# sample['Sand'] = preds[:, 4]
#
# sample.to_csv('submission.csv', index=False)

# cases = [ None, [0], [1], [2], [0, 1], [1, 2], [0, 2], [0, 1, 2]]
# layer_outs_dfs = [pd.DataFrame([1]), pd.DataFrame([11]), pd.DataFrame([111])]
# df = pd.DataFrame([2])
# def concat_data(case, layer_outs_dfs, df):
#     if case == None: return df
#     dfs = [layer_outs_dfs[c] for c in case]
#     return pd.concat(dfs + [df], axis=1)
#
# for case in cases:
#     res = concat_data(case, layer_outs_dfs, df)
#     print(res.shape, res.values)

# scores_all = [(None, {'Ca': 0.90781596086993555, 'P': 0.51837853334581763, 'pH': 0.37263845728013828, 'SOC': 1.0092687823934969, 'Sand': 0.26702602235295902}),
# ([0], {'Ca': 0.91237824166868342, 'P': 0.53771438657920623, 'pH': 0.3687357843494381, 'SOC': 1.0048212793754969, 'Sand': 0.27534528988191131}),
# ([1], {'Ca': 0.92710083084292472, 'P': 0.5300810334714755, 'pH': 0.36471889264750734, 'SOC': 0.45718165992030751, 'Sand': 0.27389082053640001}),
# ([2], {'Ca': 0.94821391320296144, 'P': 0.52350977548389444, 'pH': 0.36508545424444344, 'SOC': 1.0049564537146227, 'Sand': 0.27542062387921062}),
# ([0, 1], {'Ca': 0.9196964460491277, 'P': 0.51724370013709864, 'pH': 0.36335255523033466, 'SOC': 1.0049206067696201, 'Sand': 0.26791234665678476}),
# ([1, 2], {'Ca': 0.95363622939911741, 'P': 0.56177892346932989, 'pH': 0.36468176726298568, 'SOC': 0.68241097834410391, 'Sand': 0.27407402574397455}),
# ([0, 2], {'Ca': 0.93311830867675527, 'P': 0.53928866367952677, 'pH': 0.36521180878157578, 'SOC': 1.0041002418053613, 'Sand': 0.2779252240083489}),
# ([0, 1, 2], {'Ca': 0.93336638177564357, 'P': 0.52471195652507341, 'pH': 0.36410788132614141, 'SOC': 0.99864542927838662, 'Sand': 0.27294893776700635})]
# scores_avg = [(score[0], np.mean(list(score[1].values()))) for score in scores_all]
# print(scores_avg)
if __name__ == "__main__":
    main('Res_model', development=True)