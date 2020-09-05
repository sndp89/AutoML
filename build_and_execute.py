"""
    build_and_execute.py - EBI Data Science team home grown product to build machine learning models at scale. 
    
    This file is the entire automation module for model building. This module is available to all Comcast employees. Not restricted to just the EBI Data Science team.
    
    Core Functionality:
    
1. Performs feature selection
2. Develops machine learning models (5 different algorithms). GPU based algorithms integration under process.
3. Validates the models on hold out datasets
4. Picks the best algorithm to deploy based on user selected statistics (ROC, KS, Accuracy)
5. Produces pseudo score code for production deployment
"""

# In[1]:
from pyspark import SparkContext,HiveContext,Row,SparkConf
from pyspark.sql import *
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import *
from pyspark.mllib.stat import *
from pyspark.ml.feature import *
from pyspark.ml.feature import IndexToString,StringIndexer,VectorIndexer
from sklearn.metrics import roc_curve,auc
import numpy as np
import pandas as pd
import subprocess
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import functions as func
from datetime import *
from pyspark.sql import SparkSession,SQLContext
from pyspark.sql.types import *
from dateutil.relativedelta import relativedelta
import datetime
from datetime import date
import string
import os
import sys
import time
import numpy

spark = SparkSession.builder.appName("MLA_Automated_model").enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
sc = spark.sparkContext

import_data = False
stop_run = False
message = ''
filename = ''

# assign the parameters passed from the front end API to the program here

nt_id = sys.argv[1]
email_id = sys.argv[2]
mdl_output_id = sys.argv[3]
mdl_prefix = sys.argv[4]

dev_table_name = sys.argv[5]
oot1_table_name = sys.argv[6]
oot2_table_name = sys.argv[7]

delimiter_type = sys.argv[11]

final_vars_table = sys.argv[12] # output of MLA variable reduction 
include_vars = sys.argv[13] # user specified variables to be used
include_prefix = sys.argv[14] # user specified prefixes to be included for modeling
include_suffix = sys.argv[15] # user specified prefixes to be included for modeling
exclude_vars = sys.argv[16] # user specified variables to be excluded for modeling
exclude_prefix = sys.argv[17] # user specified prefixes to be excluded for modeling
exclude_suffix = sys.argv[18] # user specified suffixes to be excluded for modeling

target_column_name = sys.argv[19]

run_logistic_model = eval(sys.argv[20])
run_randomforest_model = eval(sys.argv[21])
run_boosting_model = eval(sys.argv[22])
run_gpu_model = eval(sys.argv[23])
run_neural_model = eval(sys.argv[24])
run_other_models = eval(sys.argv[25])

miss_per = float(sys.argv[26])
impute_with = float(sys.argv[27])
train_size=float(sys.argv[28])
valid_size=float(sys.argv[29])
seed=int(sys.argv[30])

model_selection_criteria = sys.argv[31] #possible_values ['ks','roc','accuracy']
dataset_to_use = sys.argv[32] #possible_values ['train','valid','test','oot1','oot2']

data_folder_path = sys.argv[33]
hdfs_folder_path = sys.argv[34]

# assign input files if the user uploaded files instead of tables.
if dev_table_name.strip() == '':
    dev_input_file = sys.argv[8]
    if dev_input_file.strip() == '':
        print('Please provide a development table or development file to process the application')
        stop_run = True
        message = 'Development Table or file is not provided. Please provide a development table or file name to process'
        
    import_data = True
    file_type = dev_table_name.split('.')[-1]
    out,err=subprocess.Popen(['hadoop', 'fs', '-copyFromLocal',data_folder_path+dev_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    
if oot1_table_name.strip() == '':
    oot1_input_file = sys.argv[9]
    out,err=subprocess.Popen(['hadoop', 'fs', '-copyFromLocal',data_folder_path+oot1_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

if oot2_table_name.strip() == '':
    oot2_input_file = sys.argv[10]
    out,err=subprocess.Popen(['hadoop', 'fs', '-copyFromLocal',data_folder_path+oot2_input_file,hdfs_folder_path],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()

ignore_data_type = ['timestamp', 'date']
ignore_vars_based_on_datatype = []

# extract the input variables in the file or table
if not stop_run:
    if import_data:
        df = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv('hdfs://comcasticeprod' + hdfs_folder_path + dev_input_file)
        df = pd.DataFrame(zip(*df.dtypes),['col_name', 'data_type']).T
    else:
        df = spark.sql('describe ' + dev_table_name)
        df = df.toPandas()
    
    input_vars = list(str(x.lower()) for x in df['col_name']) 
    for i in ignore_data_type:
        ignore_vars_based_on_datatype += list(str(x) for x in df[df['data_type'] == i]['col_name'])
    
    if len(ignore_vars_based_on_datatype) > 0:
        input_vars = list(set(input_vars) - set(ignore_vars_based_on_datatype))   
    
    input_vars.remove(target_column_name)
    # In[3]:

    ## variables to include
    import re
    prefix_include_vars = []
    suffix_include_vars = []

    if include_vars.strip() != '':
        include_vars = re.findall(r'\w+', include_vars.lower())

    if include_prefix.strip() != '':
        prefix_to_include = re.findall(r'\w+', include_prefix.lower())

        for i in prefix_to_include:
            temp = [x for x in input_vars if x.startswith(str(i))]
            prefix_include_vars.append(temp)

        prefix_include_vars = [item for sublist in prefix_include_vars for item in sublist]

    if include_suffix.strip() != '':
        suffix_to_include = re.findall(r'\w+', include_suffix.lower()) 

        for i in suffix_to_include:
            temp = [x for x in input_vars if x.startswith(str(i))]
            suffix_include_vars.append(temp)

        suffix_include_vars = [item for sublist in suffix_include_vars for item in sublist]

    include_list = list(set(include_vars) | set(prefix_include_vars) | set(suffix_include_vars))

    ## Variables to exclude
    prefix_exclude_vars = []
    suffix_exclude_vars = []

    if exclude_vars.strip() != '':
        exclude_vars = re.findall(r'\w+', exclude_vars.lower())

    if exclude_prefix.strip() != '':
        prefix_to_exclude = re.findall(r'\w+', exclude_prefix.lower())

        for i in prefix_to_exclude:
            temp = [x for x in input_vars if x.startswith(str(i))]
            prefix_exclude_vars.append(temp)

        prefix_exclude_vars = [item for sublist in prefix_exclude_vars for item in sublist]

    if exclude_suffix.strip() != '':
        suffix_to_exclude = re.findall(r'\w+', exclude_suffix.lower()) 

        for i in suffix_to_exclude:
            temp = [x for x in input_vars if x.startswith(str(i))]
            suffix_exclude_vars.append(temp)

        suffix_exclude_vars = [item for sublist in suffix_exclude_vars for item in sublist]

    exclude_list = list(set(exclude_vars) | set(prefix_exclude_vars) | set(suffix_exclude_vars))


    # In[4]:

    if len(include_list) > 0:
        input_vars = list(set(input_vars) & set(include_list))

    if len(exclude_list) > 0:
        input_vars = list(set(input_vars) - set(exclude_list))


    # In[5]:

    ## Variable reduction API - Reduce variable based on MLA variable reduction output tables

    if final_vars_table.strip() != '':
        try:
            df = spark.sql("select * from " + final_vars_table)
            df = df.toPandas()
            df.columns = [x.lower() for x in df.columns]
            input_vars = list(str(x) for x in df['variable'])
            input_vars = [x[:-len('_index')] if x.endswith('_index') else x for x in input_vars]
        except:
            try:
                df = spark.sql('describe ' + final_vars_table)
                df = df.toPandas()
                input_vars = list(str(x.lower()) for x in df['col_name']) 
                input_vars.remove(target_column_name)
            except:
                pass
    
    ## Variable reduction API - Invoke MLA variable reduction API if the number of the variables is more than 300
    import os      
    import subprocess
    
    if len(input_vars) > 300:
        try:
            df = spark.sql('describe database ' + nt_id);
        except:
            nt_id = 'ebiklondikep'
        
        if not os.path.exists('/home/' + nt_id + '/' + 'mla_variable_reduction'):
            os.makedirs('/home/' + nt_id + '/' + 'mla_variable_reduction')
        
        subprocess.call(['chmod', '777', '-R', '/home/' + nt_id + '/' + 'mla_variable_reduction/'])
        
        if not os.path.exists('/home/' + nt_id + '/' + 'mla_variable_reduction/logs/' + mdl_prefix + '/'):
            os.makedirs('/home/' + nt_id + '/' + 'mla_variable_reduction/logs/' + mdl_prefix + '/') 
        
        #subprocess.call(['chmod', '777', '-R', '/home/' + nt_id + '/' + 'mla_variable_reduction/'])
        log_file_path = '/home/' + nt_id + '/' + 'mla_variable_reduction/logs/' + mdl_prefix + '/'
        logging_path = '/home/' + nt_id + '/' + 'mla_variable_reduction/logs/' + mdl_prefix + '/error'

        if len(final_vars_table.strip()) == 0:
            final_vars_table = mdl_prefix + '_top_vars'

        command_to_run = string.Template("""export PYSPARK_PYTHON=/opt/cloudera/parcels/Anaconda/bin/python && /home/ebiklondikep/spark-2.4.0-bin-hadoop2.6/bin/spark-submit --master yarn --deploy-mode client --driver-memory 8g --num-executors 300 --executor-memory 8g --executor-cores 3 --name=mla_auto_model_builder --queue root.Two --conf spark.ui.showConsoleProgress=false --conf spark.driver.memoryOverhead=4096 --conf spark.executor.memoryOverhead=4096 --conf spark.yarn.access.hadoopFileSystems=hdfs://comcasticeprod --conf spark.driver.extraJavaOptions=-Dlog4jspark.root.logger=ERROR,console /home/advanl/ref/variable_reduction_app/api/vr_v1.py """ + dev_table_name.split('.')[0] + " " + dev_table_name.split('.')[1] + """ ${target_column_name} ${nt_id} ${final_vars_table} Y Y Y ${email_id} ${log_file_path} > ${logging_path} """).substitute(locals())
        print(command_to_run)

        out,err = out,err=subprocess.Popen([command_to_run],shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    
        try:
            df = spark.sql("select * from " + nt_id + '.' + final_vars_table)
            df = df.toPandas()
            df.columns = [x.lower() for x in df.columns]
            input_vars = list(str(x) for x in df['variable'])
            input_vars = [x[:-len('_index')] if x.endswith('_index') else x for x in input_vars]
        except:
            message = message + """Run the MLA Variable reduction API before running this API. Some error deducted in Variable Reduction process. Please use the link http://ebdp-avdc-d010p.sys.comcast.net:3003/ to run the API"""
            stop_run = True
     
    ## By this step, the number of variables are reduced to 300 or less
    
if not stop_run:
    variables_used_in_other_models = """
    """
    # directly from score file specification 
    id_variables = """
    klondike_primary_key
    housekey
    account
    accountid
    accountstatus
    serloc_corp_sysprin
    serloc_current_region_name
    serloc_current_division_name
    epsilon_cdm_acct_dim_eps_acct_id
    epsilon_cdm_busn_base_eps_busn_id
    decile_segment_flag
    epsilon_cdm_busn_base_eps_addr_id
    """

    tera_id_variables = """customer_account_id
    account_number"""

    add_vars = filter(None,list(variables_used_in_other_models.split('\n'))) # additinal variables to be used to train
    id_vars = filter(None,list(id_variables.split('\n')))#list of id variables
    tera_id_vars = filter(None,list(tera_id_variables.split('\n')))#list of id variables
    
    add_vars = filter(None,[x.strip() for x in add_vars]) #removes whitespaces
    id_vars = filter(None,[x.strip() for x in id_vars]) #removes whitespaces
    tera_id_vars = filter(None,[x.strip() for x in tera_id_vars]) #removes whitespaces
    input_vars = filter(None,[x.strip() for x in input_vars]) #removes whitespaces
    
    final_vars = list(set(input_vars + add_vars) - set(id_vars + tera_id_vars)) # final list of variables to be pulled

    from datetime import datetime
    insertion_date = datetime.now().strftime("%Y-%m-%d")

    import re
    from pyspark.sql.functions import col

    # remove spaces from column names
    new_cols = filter(None,[x.strip() for x in final_vars + id_vars + tera_id_vars])
    final_vars = filter(None,[x.strip() for x in final_vars])
    input_vars = filter(None,[x.strip() for x in input_vars])
    id_vars = filter(None,[x.strip() for x in id_vars])
    tera_id_vars = filter(None,[x.strip() for x in tera_id_vars])

    print('final_vars - ', final_vars)
    print('input_vars - ', input_vars)
    print('id_vars - ', id_vars)
    print('tera_id_vars - ', tera_id_vars)


    # import data for the modeling
    if import_data:
        train_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv('hdfs://comcasticeprod' + hdfs_folder_path + dev_input_file)
        oot1_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv('hdfs://comcasticeprod' + hdfs_folder_path + oot1_input_file)
        oot2_table = spark.read.option("delimiter",delimiter_type).option("header", "true").option("inferSchema", "true").csv('hdfs://comcasticeprod' + hdfs_folder_path + oot2_input_file)
    else:
        train_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + dev_table_name)
        oot1_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + oot1_table_name)
        oot2_table = spark.sql("select " + ", ".join(final_vars + [target_column_name]) + " from " + oot2_table_name)

    train_table = train_table.where(train_table[target_column_name].isNotNull())
    oot1_table = oot1_table.where(oot1_table[target_column_name].isNotNull())
    oot2_table = oot2_table.where(oot2_table[target_column_name].isNotNull())

    X_train = train_table.select(final_vars)
    X_train.cache()

    # apply data manipulations on the data - missing value check, label encoding, imputation

    from data_manipulations import *

    vars_selected_train = missing_value_calculation(X_train, miss_per) # missing value check

    # In[9]:

    vars_selected = filter(None,list(set(list(vars_selected_train))))
    X = X_train.select(vars_selected)
    X = X.cache()

    Y = train_table.select(target_column_name)
    Y = Y.cache()


    # In[10]:

    char_vars, num_vars = identify_variable_type(X)
    X, char_labels = categorical_to_index(X, char_vars) #label encoding
    X = numerical_imputation(X,num_vars, impute_with) # imputation
    X = X.select([c for c in X.columns if c not in char_vars])
    X = rename_columns(X, char_vars)
    joinedDF = join_features_and_target(X, Y)

    joinedDF = joinedDF.cache()
    print('Features and targets are joined')

    train, valid, test = train_valid_test_split(joinedDF, train_size, valid_size, seed)
    train = train.cache()
    valid = valid.cache()
    test = test.cache()
    print('Train, valid and test dataset created')


    # In[11]:

    x = train.columns
    x.remove(target_column_name)
    feature_count = len(x)
    print(feature_count)

    if feature_count > 30:
        print('# No of features - ' + str(feature_count) + '.,  Performing feature reduction before running the model.')
    
    # directory to produce the outputs of the automation
    import os
    
    try:
        if not os.path.exists('/home/' + nt_id + '/' + 'mla_' + mdl_prefix):
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix)
    except:
        nt_id = 'ebiklondikep'
        if not os.path.exists('/home/' + nt_id + '/' + 'mla_' + mdl_prefix):
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix)
    
    subprocess.call(['chmod','777','-R','/home/' + nt_id + '/' + 'mla_' + mdl_prefix])
    
    x = train.columns
    x.remove(target_column_name)
    sel_train = assembled_vectors(train,x, target_column_name)
    sel_train.cache()

    # # Variable Reduction for more than 30 variables in the feature set using Random Forest

    # In[12]:

    from pyspark.ml.classification import  RandomForestClassifier
    from feature_selection import *

    rf = RandomForestClassifier(featuresCol="features",labelCol = target_column_name)
    mod = rf.fit(sel_train)
    varlist = ExtractFeatureImp(mod.featureImportances, sel_train, "features")
    selected_vars = [str(x) for x in varlist['name'][0:30]]
    train = train.select([target_column_name] + selected_vars)
    train.cache()
    
    save_feature_importance(nt_id, mdl_prefix, varlist) #Create feature importance plot and excel data
    
    x = train.columns
    x.remove(target_column_name)
    feature_count = len(x)
    print(feature_count)

    train, valid, test, pipelineModel = scaled_dataframes(train,valid,test,x,target_column_name)

    train = train.cache()
    valid = valid.cache()
    test = test.cache()
    print('Train, valid and test are scaled')


    # import packages to perform model building, validation and plots

    import time
    from validation_and_plots import *
    
    #apply the transformation done on train dataset to OOT 1 and OOT 2 using the score_new_df function
    def score_new_df(scoredf):
        newX = scoredf.select(final_vars)
        #idX = scoredf.select(id_vars)

        newX = newX.select(list(vars_selected))
        newX = char_labels.transform(newX)
        newX = numerical_imputation(newX,num_vars, impute_with)
        newX = newX.select([c for c in newX.columns if c not in char_vars])
        newX = rename_columns(newX, char_vars)

        finalscoreDF = pipelineModel.transform(newX)
        finalscoreDF.cache()
        return finalscoreDF

    #apply the transformation done on train dataset to OOT 1 and OOT 2 using the score_new_df function
    
    x = 'features'
    y = target_column_name

    oot1_targetY = oot1_table.select(target_column_name)
    oot1_intDF = score_new_df(oot1_table)
    oot1_finalDF = join_features_and_target(oot1_intDF, oot1_targetY)
    oot1_finalDF.cache()
    print(oot1_finalDF.dtypes)

    oot2_targetY = oot2_table.select(target_column_name)
    oot2_intDF = score_new_df(oot2_table)
    oot2_finalDF = join_features_and_target(oot2_intDF, oot2_targetY)
    oot2_finalDF.cache()
    print(oot2_finalDF.dtypes)

    # run individual models

    from model_builder import *
    from metrics_calculator import *

    KerasModel = ''
    loader_model_list = []    
    dataset_list = ['train','valid','test','oot1','oot2']
    datasets = [train,valid,test,oot1_finalDF, oot2_finalDF]
    models_to_run = []

    if run_logistic_model:
        lrModel = logistic_model(train, x, y) #build model
        lrModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/logistic_model.h5') #save model object
        print("Logistic model developed")
        model_type = 'Logistic'
        l = []

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(nt_id, mdl_prefix, i, y, lrModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(nt_id, mdl_prefix, model_type) #ks charts
        joblib.dump(l,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix  + '/logistic_metrics.z') #save model metrics
        models_to_run.append('logistic')
        loader_model_list.append(LogisticRegressionModel)

    if run_randomforest_model:
        rfModel = randomForest_model(train, x, y) #build model
        rfModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/randomForest_model.h5') #save model object
        print("Random Forest model developed")
        model_type = 'RandomForest'
        l = []

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(nt_id, mdl_prefix, i, y, rfModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(nt_id, mdl_prefix, model_type) #ks charts
        joblib.dump(l,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/randomForest_metrics.z') #save model metrics
        models_to_run.append('randomForest')
        loader_model_list.append(RandomForestClassificationModel)

    if run_boosting_model:
        gbModel = gradientBoosting_model(train, x, y) #build model
        gbModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/gradientBoosting_model.h5') #save model object
        print("Gradient Boosting model developed")
        model_type = 'GradientBoosting'
        l = []

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(nt_id, mdl_prefix, i, y, gbModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(nt_id, mdl_prefix, model_type) #ks charts
        joblib.dump(l,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/gradientBoosting_metrics.z') #save model metrics
        models_to_run.append('gradientBoosting')
        loader_model_list.append(GBTClassificationModel)

    if run_other_models:
        dtModel = decisionTree_model(train, x, y) #build model
        dtModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/decisionTree_model.h5') #save model object
        print("Decision Tree model developed")
        model_type = 'DecisionTree'
        l = []

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(nt_id, mdl_prefix, i, y, dtModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(nt_id, mdl_prefix, model_type) #ks charts
        joblib.dump(l,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/decisionTree_metrics.z') #save model metrics
        models_to_run.append('decisionTree')
        loader_model_list.append(DecisionTreeClassificationModel)

    if run_neural_model:
        mlpModel = neuralNetwork_model(train, x, y, feature_count) #build model
        mlpModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/neuralNetwork_model.h5') #save model object
        print("Neural Network model developed")
        model_type = 'NeuralNetwork'
        l = []

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass

        for i in datasets:
            l += model_validation(nt_id, mdl_prefix, i, y, mlpModel, model_type, dataset_list[datasets.index(i)]) #validate model

        draw_ks_plot(nt_id, mdl_prefix, model_type) #ks charts
        joblib.dump(l,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/neuralNetwork_metrics.z') #save model metrics
        models_to_run.append('neuralNetwork')
        loader_model_list.append(MultilayerPerceptronClassificationModel)

    if run_gpu_model:
        models_to_run.append('keras')
        loader_model_list.append(KerasModel)
        model_type = 'Keras'

        try:    
            os.mkdir('/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + str(model_type))
        except:
            pass
    
    # model building complete. Let us validate the metrics for the models created
    
    
    # model validation part starts now.
    from model_selection import *
    output_results = select_model(nt_id, mdl_prefix, model_selection_criteria, dataset_to_use) #select Champion, Challenger based on the metrics provided by user

    #print(type(output_results), output_results)

    selected_model = output_results['model_type'][0] #Champion model based on selected metric

    load_model = loader_model_list[models_to_run.index(selected_model)] #load the model object for Champion model
    model = load_model.load('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/' + selected_model + '_model.h5')

    print('Model selected for scoring - ' + selected_model)

    # In[21]:
    
    # Produce pseudo score for production deployment
    # save objects produced in the steps above for future scoring
    from sklearn.externals import joblib

    char_labels.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/char_label_model.h5')
    pipelineModel.write().overwrite().save('/user/' + nt_id + '/' + 'mla_' + mdl_prefix + '/pipelineModel.h5')

    save_list = [final_vars,id_vars,vars_selected,char_vars,num_vars,impute_with,selected_model,dev_table_name]
    joblib.dump(save_list,'/home/' + nt_id + '/' + 'mla_' + mdl_prefix + '/model_scoring_info.z')


    # # Create score code

    # In[22]:
    
    from scorecode_creator import *
    selected_model_scorecode(nt_id, mdl_output_id, mdl_prefix, parameters) 
    individual_model_scorecode(nt_id, mdl_output_id, mdl_prefix, parameters)
    
    message = message + 'Model building activity complete and the results are attached with this email. Have Fun'
    
    from zipper_function import *
    try:       
        filename = zipper('/home/' + nt_id + '/' + 'mla_' + mdl_prefix)
    except:
        filename = ''
    
# clean up files loaded in the local path
if import_data:
    file_list = [dev_input_file, oot1_input_file, oot2_input_file]

    for i in list(set(file_list)):
        try:
            os.remove(data_folder_path + str(i))
        except:
            pass
    
# clean up files loaded in the hdfs path
if import_data:
    file_list = [dev_input_file, oot1_input_file, oot2_input_file]
    
    for i in list(set(file_list)):
        try:
            out,err=subprocess.Popen(['hadoop', 'fs', '-rm','-r','-f',hdfs_folder_path+str(i)],stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
        except:
            pass

# send the final email confirmation to users
from followup_email import *
emailer(nt_id, email_id, mdl_prefix, [filename], message)

