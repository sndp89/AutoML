#!/usr/bin/env python
# coding: utf-8
"""
Keras - Automated Model building
This code is a integral part of MLA Automated model builder. It is customized for MLA.
Feature pre processing and scaling of variables are done before invoking this code.
If you need to run as is, then it will only accept numeric variables.

Datasets required to run this code - Train, Valid, Test, OOT1 and OOT2.

To run this code from shell use the following

python 
"""

# Import packages
import os
import sys
import string
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from shutil import copyfile
from gpu_model_validation import *
from create_keras_score_code import *
from zipper_function import *

# Assign system arguments
train_filename = sys.argv[1]
valid_filename = sys.argv[2]
test_filename = sys.argv[3]
oot1_filename = sys.argv[4]
oot2_filename = sys.argv[5]
target_column_name = sys.argv[6]
model_prefix = sys.argv[7]
oot1_oot2_same = sys.argv[8]

# Do not alter the codes below. Do at your own risk.

# Keras model filenames and paths assignment
model_type = 'Keras'
metrics_filename = 'keras_metrics.xlsx'
model_filename = 'keras_model.h5'
path_to_read = '/my-ml-files/data'
path_to_save = '/my-ml-files/output_results'

# List of files and data to process
list_of_datatypes = ['train','valid','test','oot1']
list_of_files = [train_filename,valid_filename,test_filename,oot1_filename]

if not oot1_oot2_same:
    list_of_datatypes += ['oot2']
    list_of_files += [oot2_filename]

# Create output directory for saving the output files
if not os.path.exists(str(path_to_save) + '/' + str(model_prefix)):
    os.makedirs(str(path_to_save) + '/' + str(model_prefix))

# Keras model architecture for binary targets. Fixed model architecture.

def keras_model(features, target, epochs=20,batch_size=512,validation_split=0.2):
    
    model = Sequential()

    model.add(Dense(64, activation='relu', input_shape=(features.shape[1],)))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    early_stopping_monitor = EarlyStopping(patience=3)

    model.fit(features, target, epochs=epochs, batch_size=batch_size, verbose=1, \
              callbacks=[early_stopping_monitor], validation_split=validation_split)
    return model

# score data and produce deciles
def scoring(features,target,clf):
    
    score = pd.DataFrame(clf.predict_proba(features), columns = ['SCORE'])
    score['DECILE'] = pd.qcut(score['SCORE'].rank(method = 'first'),10,labels=list(range(10,0,-1)))
    score['DECILE'] = score['DECILE'].astype(float)
    score['TARGET'] = np.array(target)
    score['NONTARGET'] = 1 - score['TARGET']
    score['PREDICT'] = clf.predict_classes(features)
    return score

# call the scoring function. Once the data is score, it performs validation on the scored data. 
# Returns ROC, Accuracy and KS
def score_and_validate(path_to_read, path_to_save, model_prefix, model_type, data_type, file_name, target_column_name, model):
    
    l = []
    df = pd.read_csv(str(path_to_read) + '/' + str(model_prefix) + '/' + file_name, sep='|')
    features = df[df.columns.difference([target_column_name])]
    y = df[target_column_name]
    
    scored_data = scoring(features, y, model)
    ks = deciling(scored_data, ['DECILE'], 'TARGET', 'NONTARGET', path_to_save, \
                  model_prefix, model_type, data_type)
    roc = calculate_roc(scored_data['SCORE'], scored_data['TARGET'], path_to_save, \
                        model_prefix, model_type, data_type)
    accuracy = draw_confusion_matrix(scored_data['PREDICT'], scored_data['TARGET'], \
                                     path_to_save, model_prefix, model_type, data_type)
    
    l = [roc, accuracy, ks]
    del df, features, y
    return l

## Main function

if __name__== "__main__":
    
    df = pd.read_csv(str(path_to_read) + '/' + str(model_prefix) + '/' + list_of_files[0], sep='|') #read train data
    X_train = df[df.columns.difference([target_column_name])] # split features from train
    Y_train = df[target_column_name] #split target from train

    model = keras_model(X_train, Y_train) #build Keras model
    print("Keras model building is complete. Still performing Validation")
    
    # validate Keras model and collect stats
    results = []
    for i in list_of_files:
        results += score_and_validate(path_to_read, path_to_save, model_prefix, model_type, list_of_datatypes[list_of_files.index(i)], i, target_column_name, model)
        
    # If oot2 is same as oot1, then assign oot1 results to oot2 without doing any processing.
    if oot1_oot2_same:
        results += results[-3:]
        copyfile(str(path_to_save) + '/' + str(model_prefix) + '/' + str(model_type) + \
                 ' Model - ROC for oot1 data.png', str(path_to_save) + '/' + str(model_prefix) + \
                 '/' + str(model_type) + ' Model - ROC for oot2 data.png')
        copyfile(str(path_to_save) + '/' + str(model_prefix) + '/' + str(model_type) + \
                 ' Model - Confusion Matrix for oot1 data.png', str(path_to_save) + '/' + \
                 str(model_prefix) + '/' + str(model_type) + ' Model - Confusion Matrix for oot2 data.png')
        copyfile(str(path_to_save) + '/' + str(model_prefix) + '/' + 'KS ' + str(model_type) + \
                 ' Model oot1.xlsx', str(path_to_save) + '/' + str(model_prefix) + '/' + 'KS ' \
                 + str(model_type) + ' Model oot2.xlsx')
    
    draw_ks_plot(path_to_save, model_prefix, model_type) #create the KS excel
    print("Keras validation complete.")

    # create the final metrics excel
    final_results = pd.DataFrame({},columns=['roc_train', 'accuracy_train', 'ks_train', \
                                             'roc_valid', 'accuracy_valid', 'ks_valid', \
                                             'roc_test', 'accuracy_test', 'ks_test', \
                                             'roc_oot1', 'accuracy_oot1', 'ks_oot1', \
                                             'roc_oot2', 'accuracy_oot2', 'ks_oot2'])
    final_results.loc[str(model_type)] = results
    final_results.index = final_results.index.set_names(['model_type'])
    final_results.reset_index(inplace=True)
    final_results.to_excel(str(path_to_save) + '/' + str(model_prefix) + '/' + str(metrics_filename))
    
    # save keras model
    model.save(str(path_to_save) + '/' + str(model_prefix) + '/' + str(model_filename))
    
    # create a psuedo keras code
    keras_score_code_generator(path_to_save, model_prefix, model_filename)
    
    try:       
        filename = zipper(str(path_to_save) + '/' + str(model_prefix))
    except:
        filename = ''

    