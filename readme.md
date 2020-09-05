# Automated Model Builder API

EBI Data science team home grown product to build machine learning models at scale. Available to Comcast employees. Not restricted to just the EBI Data Science team.

## 1. Core Functionality:

1. Performs feature selection
2. Develops machine learning models (5 different algorithms). GPU based algorithms integration under process.
3. Validates the models on hold out datasets
4. Picks the best algorithm to deploy based on user selected statistics (ROC, KS, Accuracy)
5. Produces pseudo score code for production deployment

Link to run the API - http://ebdp-avdc-d010p.sys.comcast.net:3010/

![](https://github.comcast.com/ebi-modeling/MLA/blob/development/auto_model_builder/automated_model_building_api.png)

## 2. Features included in the automation
### 2.1 Supported input data types

ICE hive tables (or) CSV or TXT files. CSV and TXT files can be uploaded directly from local laptop.

### 2.2 Variable Reduction MLA integration

MLA variable reduction API is currently integrated with automated model builder. The users can directly run this module after you have generated the datasets using Data pull API.

### 2.3 Algorithms currently integrated

1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. Neural Network
5. Decision Tree
6. Keras Model integration under process

### 2.4 Supported target types 

1. Binary

### 2.5 Output Stats

1. KS charts
2. ROC charts
3. Confusion Matrix plot. This also includes the following stats - Accuracy, Precision, Recall and F1 Score

### 2.6 Output files

1. Metrics.xlsx - This is a excel file with all the collected stats and also points to Champion and Challenger model
2. Pseudo score code for production deployment based on selected algorithms for model building

Please reach out to EBI Data Science team for any questions or support. If you are interested in adding modules, please reach out to Josh (Joshua_Berry2@cable.comcast.com). 

## 3. Testing

Here's the best steps to take in order to test and debug this code. Use edge node - `ebdp-avdc-d010p.sys.comcast.net`. (We found that Python Styler function has some issues in other edge nodes except d010p. So please use only d010p to perform your testing).

1. Create your own branch, download it, extract it, and put it in ICE somewhere (Preferably your home folder path - 
`/home/nt_id/MLA`)

2. Open the utils/const.py file and do the following  
  2.1 update the repo path to point to your new branch (`/home/nt_id/MLA`)  
  2.2 change debug = True to execute in Test mode. You wont be able to execute and test your new script, if you did not change this option to True.
  2.3 Note the testing port (or change it).   

... recode ...

Navigate to your branch and execute this to test the form:
```sh
cd auto_model_builder
python Form/webform_application.py
```
Here is some code to test the script executor, if you are making __no changes to the front end API__ (Any element in the `Form` folder).

```sh
export folder_path='/home/nt_id/MLA' #note this path is similar to path where you cloned the MLA repository
export nt_id='mbagav200' #your nt_id
export email_id='sundar_krishnan@cable.comcast.com' #your email id
export model_output_id='EBI_N19_06_999' #your model_output_id
export model_prefix='SUB_MODEL_TEST' #your model_prefix
export dev_table='mbagav200.rnps_dev'
export oot1_table='mbagav200.rnps_oot1' #populate the same as dev table, if you dont have the oot1 table 
export oot2_table='mbagav200.rnps_oot2' #populate the same as dev table, if you dont have the oot2 table
export target='target' #target column name
export log_file_path='/home/nt_id/log_file.log'

# do not change the script below
export PYSPARK_PYTHON=/opt/cloudera/parcels/Anaconda/bin/python 
/home/ebiklondikep/spark-2.4.0-bin-hadoop2.6/bin/spark-submit --master yarn --deploy-mode client --driver-memory 8g --num-executors 300 --executor-memory 8g --executor-cores 3 --name=mla_auto_model_builder --queue root.Two --conf spark.ui.showConsoleProgress=false --conf spark.yarn.driver.memoryOverhead=4096 --conf spark.yarn.executor.memoryOverhead=4096 --conf spark.yarn.access.hadoopFileSystems=hdfs://comcasticeprod --conf spark.driver.extraJavaOptions=-Dlog4jspark.root.logger=ERROR,console $folder_path/auto_model_builder/build_and_execute.py $nt_id $email_id $model_output_id $model_prefix $dev_table $oot1_table $oot2_table '' '' '' '' '' '' '' '' '' '' '' $target 'True' 'True' 'False' 'False' 'False' 'False' '0.75' '0' '0.4' '0.3' '12345' 'ks' 'train' '/home/mbagav200/auto_model_builder/test_data/' '/user/mbagav200/' > $log_file_path &
```
## 4. Git option

Currently in ICE the git clone option is available in the following edge nodes - `ebdp-avdc-d001p.sys.comcast.net,ebdp-avdc-d002p.sys.comcast.net,ebdp-avdc-d003p.sys.comcast.net,ebdp-avdc-d004p.sys.comcast.net,ebdp-avdc-d005p.sys.comcast.net`. We recommend using `ebdp-avdc-d005p.sys.comcast.net` for git operations.

Please use the following codes to perform git clone, then switch to `ebdp-avdc-d010p.sys.comcast.net` edge node to continue your testing. (We found that Python Styler function has some issues in other edge nodes except d010p. So please use only d010p to perform your testing).

All the below commands should be run from the server d005p terminal.

### 4.1 Setting up configuration names to be used for your project
```
git config user.name "mbagav200"
git config user.email "sundar_krishnan@cable.comcast.com"
git config --list
```

### 4.2 First time git user
```
ssh-keygen -t rsa
vi /home/nt_id/.ssh/id_rsa.pub #use your nt_id
```
The above command should have generated a public RSA token. Please copy the contents of the token and paste in the link below.  
  
https://github.comcast.com/settings/ssh/new  
  
Once you have copied the token, please execute the following command in terminal to make sure it works.  
   
```
ssh -T git@github.comcast.com
```
It should say "Hi nt_id! You've successfully authenticated, but GitHub does not provide shell access.". If it does not produce the output then execute the command below to generate the verbose to debug the issue.

```
ssh -vvvT git@github.comcast.com
```
If you still face connection issue, please reach out Anusha (Anusha_Lingareddy@comcast.com) or Venkat (Venkata_Gunnu@comcast.com) and they should be able to assist.

Once the SSH setup is successful, you can perform git operations.

### 4.3 Git clone

```
git clone -b development git@github.comcast.com:ebi-modeling/MLA.git
```

Now you should see the MLA repository in your home folder ready to be cloned. Now switch to d010p edge node to make changes and test your changes. Once testing is completed you can come back to d005p edge node to push your changes.


### 4.4 Git Remote, Push, Pull and all other stuff

```
git remote add your_git_repo_name git@github.comcast.com:ebi-modeling/MLA.git #Create your remote git repository
git remote -v                                                                 #List all your remote repository
git fetch your_git_repo_name                                                  #Fetches the files and directory
git branch -r                                                                 #List all the branch in the repo
git checkout development                                                      #Go to a specific branch
git pull your_git_repo_name development                                       #download and merge the changes in a branch
git commit -m "My updated to development branch"                              #commit the changes in a branch
git push your_git_repo_name                                                   #push the changes to github
```



