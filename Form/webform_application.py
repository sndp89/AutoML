"""
    webform_application.py
    
    This file is the middle layer between the front end API and the backend end Spark application. It does the following things
    
    1. Renders the web API
    2. Gets the arguments from the front end API and then creates the spark job run command
    3. Executes the spark job and informs the user via email
    
"""

from flask import *
import os, string
from werkzeug.utils import secure_filename
import traceback
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from followup_email import *

from utils.const import repo_path, port_automl, port_testing, debug

repo_path = repo_path + '/auto_model_builder/'

UPLOAD_FOLDER = '/home/mbagav200/auto_model_builder/test_data/' # This path is going to be constant
HDFS_UPLOAD_FOLDER = '/user/mbagav200/' # This path is going to be constant
ALLOWED_EXTENSIONS = {'txt', 'csv'}

## This is a flask application. The flask connector will run the application through the desired port and also serves the data to the backend application.
## to test, python auto_model_builder/Form/webform_application.py
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#The code below renders the front end API
@app.route('/')
def index():
    return render_template('mainform.html')

#The code below is used to upload the files in the front end to specified UPLOAD_FOLDER path provided above.
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    
    if request.method == 'POST':
        print("In uploader")
        print(str(list(request.files.to_dict().keys())[0]))
        file = request.files[str(list(request.files.to_dict().keys())[0])]
        try:
            filename = os.path.join(app.config['UPLOAD_FOLDER'],secure_filename(file.filename))
            print(file)
         
            file.save(filename)
            print("File saved to :" + filename)
        except Exception:
            traceback.print_exc()
        return 'file uploaded successfully'

# This is the main form page, where the inputs are parsed from front end and it runs the backend application.    
@app.route('/hello', methods=['POST'])        
def hello():

# user id and model id request
    print(os.getcwd())
    NT_ID = request.form['NT_ID']
    EMAIL_ID = request.form['EMAIL_ID']
    MODEL_OUTPUT_ID = request.form['MODEL_OUTPUT_ID']
    MODEL_PREFIX = request.form['MODEL_PREFIX']
    
# input tables to operate
    
    DEV_TABLE_NAME = request.form['DEV_TABLE_NAME']
    OOT1_TABLE_NAME = request.form['OOT1_TABLE_NAME']
    if OOT1_TABLE_NAME.strip() == '':
        OOT1_TABLE_NAME = DEV_TABLE_NAME
    OOT2_TABLE_NAME = request.form['OOT2_TABLE_NAME']
    if OOT2_TABLE_NAME.strip() == '':
        OOT2_TABLE_NAME = DEV_TABLE_NAME
        
# input files to operate
    
    INPUT_FILE=''
    #Created by Kruthika
    INPUT_FILE_DEV_new = ''
    INPUT_FILE_OOT1_new = ''
    INPUT_FILE_OOT2_new = ''
    
    try:
        INPUT_FILE = request.form['File_name']
    except:
        INPUT_FILE = ''

    INPUT_FILES_SPLIT = INPUT_FILE.split(';')
    print(INPUT_FILES_SPLIT)
    if(len(INPUT_FILES_SPLIT)==4):
        INPUT_FILE_DEV_new = INPUT_FILES_SPLIT[1]
        INPUT_FILE_OOT1_new = INPUT_FILES_SPLIT[2]
        INPUT_FILE_OOT2_new = INPUT_FILES_SPLIT[3]
    elif(len(INPUT_FILES_SPLIT)==3):
        INPUT_FILE_DEV_new = INPUT_FILES_SPLIT[1]
        INPUT_FILE_OOT1_new = INPUT_FILES_SPLIT[2]
        INPUT_FILE_OOT2_new = INPUT_FILES_SPLIT[1]
    elif(len(INPUT_FILES_SPLIT)==2):
        INPUT_FILE_DEV_new = INPUT_FILES_SPLIT[1]
        INPUT_FILE_OOT1_new = INPUT_FILES_SPLIT[1]
        INPUT_FILE_OOT2_new = INPUT_FILES_SPLIT[1]
        
    DELIMITER_TYPE = request.form['DELIMITER_TYPE']
    
# choose a method for variable reduction

    FINAL_VARS_TABLE = request.form['FINAL_VARS_TABLE']
    
    INCLUDE_VARS = request.form['INCLUDE_VARS']
    INCLUDE_PREFIX = request.form['INCLUDE_PREFIX']
    INCLUDE_SUFFIX = request.form['INCLUDE_SUFFIX']
    EXCLUDE_VARS = request.form['EXCLUDE_VARS']
    EXCLUDE_PREFIX = request.form['EXCLUDE_PREFIX']
    EXCLUDE_SUFFIX = request.form['EXCLUDE_SUFFIX']
    
# provide target column name
    TARGET_COLUMN_NAME = request.form['TARGET_COLUMN_NAME']
     
# models to run
    
    RUN_LOGISTIC_MODEL = request.form.get('RUN_LOGISTIC_MODEL',False)
    RUN_RANDOMFOREST_MODEL = request.form.get('RUN_RANDOMFOREST_MODEL',False)
    RUN_BOOSTING_MODEL = request.form.get('RUN_BOOSTING_MODEL', False)
    RUN_GPU_MODEL = request.form.get('RUN_GPU_MODEL',False)
    RUN_NEURAL_MODEL = request.form.get('RUN_NEURAL_MODEL',False)
    RUN_OTHER_MODEL = request.form.get('RUN_OTHER_MODEL',False)    
           
# other variables
    
    MISS_PER = 0.75
    NUMERICAL_IMPUTAION_VALUE = 0
    TRAIN_SIZE = 0.4
    VALID_SIZE = 0.3
    SEED = 12345
    
    SELECTION_CRITERIA = request.form['SELECTION_CRITERIA']
    if SELECTION_CRITERIA.strip() == '':
        SELECTION_CRITERIA = 'ks'
    DATASET_TO_USE = request.form['DATASET_TO_USE']	
    if DATASET_TO_USE.strip() == '':
        DATASET_TO_USE = 'train'

# Creates the log path for error files to display. When the user does have access to ICE default ID is set to 'mbagav200'

    try:
        if not os.path.exists('/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/logs/' + MODEL_PREFIX + '/'):
            os.makedirs('/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/logs/' + MODEL_PREFIX + '/') 
    except:
        NT_ID = 'ebiklondikep'
        if not os.path.exists('/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/logs/' + MODEL_PREFIX + '/'):
            os.makedirs('/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/logs/' + MODEL_PREFIX + '/')
    
    subprocess.call(['chmod', '777', '-R', '/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/'])
    logging_file = '/home/' + NT_ID + '/' + 'MLA_Automated_Modeling/logs/' + MODEL_PREFIX + '/error'

# Pyspark batch script generator

    string_to_run = string.Template("""sudo -H -u ${NT_ID} bash -c 'export PYSPARK_PYTHON=/opt/cloudera/parcels/Anaconda/bin/python && /home/ebiklondikep/spark-2.4.0-bin-hadoop2.6/bin/spark-submit --master yarn --deploy-mode client --driver-memory 12g --num-executors 300 --executor-memory 12g --executor-cores 3 --name=mla_auto_model_builder --queue root.Two --conf spark.ui.showConsoleProgress=false --conf spark.yarn.driver.memoryOverhead=4096 --conf spark.yarn.executor.memoryOverhead=4096 --conf spark.yarn.access.hadoopFileSystems=hdfs://comcasticeprod --conf spark.driver.extraJavaOptions=-Dlog4jspark.root.logger=ERROR,console """ + repo_path + """build_and_execute.py "${NT_ID}" "${EMAIL_ID}" "${MODEL_OUTPUT_ID}" "${MODEL_PREFIX}" "${DEV_TABLE_NAME}" "${OOT1_TABLE_NAME}" "${OOT2_TABLE_NAME}" "${INPUT_FILE_DEV_new}" "${INPUT_FILE_OOT1_new}" "${INPUT_FILE_OOT2_new}" "${DELIMITER_TYPE}" "${FINAL_VARS_TABLE}" "${INCLUDE_VARS}" "${INCLUDE_PREFIX}" "${INCLUDE_SUFFIX}" "${EXCLUDE_VARS}" "${EXCLUDE_PREFIX}" "${EXCLUDE_SUFFIX}" "${TARGET_COLUMN_NAME}" "${RUN_LOGISTIC_MODEL}" "${RUN_RANDOMFOREST_MODEL}" "${RUN_BOOSTING_MODEL}" "${RUN_GPU_MODEL}" "${RUN_NEURAL_MODEL}" "${RUN_OTHER_MODEL}" "${MISS_PER}" "${NUMERICAL_IMPUTAION_VALUE}" "${TRAIN_SIZE}" "${VALID_SIZE}" "${SEED}" "${SELECTION_CRITERIA}" "${DATASET_TO_USE}" """ + '"' + UPLOAD_FOLDER + '" "' + HDFS_UPLOAD_FOLDER + '" > "${logging_file}" &' + "'").substitute(locals())
    
    print('\n[WEBFORM] EXECUTING: \n' + str(string_to_run))
    os.system(string_to_run) # run the pyspark script in batch mode

# email functionality to inform user about the job start    
    filename = ''
    message = 'The job is being attempted. Please check the folder "/home/%s/mla_%s" in ICE for the automation output. Next, check Resource Manager to ensure that the job is running. Your model building activity has started. Once the results are available, we will attempt to send an email to %s , however the email service sometimes does not work. You can check for the error logs in the file path "/home/%s/MLA_Automated_Modeling/logs/%s/error"' % (NT_ID, MODEL_PREFIX, EMAIL_ID, NT_ID, MODEL_PREFIX) 
    emailer(NT_ID, EMAIL_ID, MODEL_PREFIX, [filename], message) # sends email to user
    
    return 'Hi %s. The job is being attempted. Please check your user folder for the script locations. Next, check Resource Manager to ensure that the job is running. Your model building activity has started. Once the results are available, we will attempt to send an email to %s , however the email service sometimes does not work. <br/> <a href="/">Back Home</a>' % (NT_ID, EMAIL_ID)

if __name__ == '__main__':
    if debug==True:
        app.run(debug=True, host = '0.0.0.0', port = port_testing)
    else:
        app.run(host = '0.0.0.0', port = port_automl)
