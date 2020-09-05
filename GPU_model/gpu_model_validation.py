import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import glob
import os
from pandas import ExcelWriter
import pandas as pd
import numpy as np
import seaborn as sns

def draw_roc_plot(fpr, tpr, roc_auc, path_to_save, model_prefix, model_type, data_type):

    plt.title(str(model_type) + ' Model - ROC for ' + str(data_type) + ' data' )
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.legend(loc = 'lower right')
    plt.savefig(str(path_to_save) + '/' + str(model_prefix) + '/' + str(model_type) + ' Model - ROC for ' + str(data_type) + ' data.png', bbox_inches='tight')
    plt.close()

def calculate_roc(y_pred, y_true, path_to_save, model_prefix, model_type, data_type):
    
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred)
    roc = metrics.auc(fpr, tpr)
    draw_roc_plot(fpr, tpr, roc, path_to_save, model_prefix, model_type, data_type)
    return roc

def deciling(data, decile_by, target, nontarget, path_to_save, model_prefix, model_type, data_type):
    
    inputs = list(decile_by)
    inputs.extend((target,nontarget))
    decile = data[inputs]
    grouped = decile.groupby(decile_by)
    agg1 = pd.DataFrame({},index=[])
    agg1['TOTAL'] = grouped.sum()[nontarget] + grouped.sum()[target]
    agg1['TARGET'] = grouped.sum()[target]
    agg1['NONTARGET'] = grouped.sum()[nontarget]
    agg1['PCT_TAR'] = grouped.mean()[target]*100
    agg1['CUM_TAR'] = grouped.sum()[target].cumsum()
    agg1['CUM_NONTAR'] = grouped.sum()[nontarget].cumsum()
    agg1['DIST_TAR'] = agg1['CUM_TAR']/agg1['TARGET'].sum()*100
    agg1['DIST_NONTAR'] = agg1['CUM_NONTAR']/agg1['NONTARGET'].sum()*100
    agg1['SPREAD'] = (agg1['DIST_TAR'] - agg1['DIST_NONTAR'])
    KS = agg1['SPREAD'].max()
    agg1.reset_index(inplace=True)
    agg1.to_excel(str(path_to_save) + '/' + str(model_prefix) + '/' + 'KS ' + str(model_type) + ' Model ' + str(data_type) + '.xlsx',index=False)
    return KS

def highlight_max(data, color='yellow'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.max()
        return [attr if v else '' for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(np.where(is_max, attr, ''),index=data.index, columns=data.columns)

def draw_ks_plot(path_to_save, model_prefix, model_type):

    writer = ExcelWriter(str(path_to_save) + '/' + str(model_prefix) + '/' + 'KS_Charts.xlsx')
    for filename in glob.glob(str(path_to_save) + '/' + str(model_prefix) + '/' + 'KS ' + str(model_type) + ' Model*.xlsx'):
        excel_file = pd.ExcelFile(filename)
        (_, f_name) = os.path.split(filename)
        (f_short_name, _) = os.path.splitext(f_name)
        for sheet_name in excel_file.sheet_names:
            df_excel = pd.read_excel(filename, sheet_name=sheet_name)
            df_excel = df_excel.style.apply(highlight_max, subset=['SPREAD'], color='#e6b71e')
            df_excel.to_excel(writer, f_short_name, index=False)
            worksheet = writer.sheets[f_short_name]
            worksheet.conditional_format('C2:C11', {'type': 'data_bar','bar_color': '#34b5d9'})#,'bar_solid': True
            worksheet.conditional_format('E2:E11', {'type': 'data_bar','bar_color': '#366fff'})#,'bar_solid': True
        os.remove(filename)
    writer.save()
    return None
    
def draw_confusion_matrix(y_pred, y_true, path_to_save, model_prefix, model_type, data_type):

    AccuracyValue =  metrics.accuracy_score(y_pred=y_pred, y_true=y_true)
    PrecisionValue = metrics.precision_score(y_pred=y_pred, y_true=y_true)
    RecallValue = metrics.recall_score(y_pred=y_pred, y_true=y_true)
    F1Value = metrics.f1_score(y_pred=y_pred, y_true=y_true)
    
    plt.title(str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data \n \n Accuracy:{0:.3f}   Precision:{1:.3f}   Recall:{2:.3f}   F1 Score:{3:.3f}\n'.format(AccuracyValue, PrecisionValue, RecallValue, F1Value))
    cm = metrics.confusion_matrix(y_true=y_true,y_pred=y_pred)
    sns.heatmap(cm, annot=True, fmt='g'); #annot=True to annotate cells
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    #print(str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png')
    plt.savefig(str(path_to_save) + '/' + str(model_prefix) + '/' + str(model_type) + ' Model - Confusion Matrix for ' + str(data_type) + ' data.png', bbox_inches='tight')
    plt.close()
    return AccuracyValue