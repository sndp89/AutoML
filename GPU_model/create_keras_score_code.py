"""
This code is used to generate a psuedo score code for Keras model
"""
import string

def keras_score_code_generator(path_to_save, model_prefix, model_filename):
    
    parameters = string.Template("""
from keras.models import load_model

model = load_model(str('${path_to_save}') + '/' + str('${model_prefix}') + '/' + str('${model_filename}'))

score_data = pd.read_csv('score_file_path.csv',sep='|') #Note this file will not have a target and it is preprocessed
scores = pd.DataFrame(model.predict_proba(score_data), columns = ['SCORE'])
scores['PREDICT'] = model.predict_classes(score_data)
""")
    parameters = parameters.substitute(locals())
    scorefile = open(str(path_to_save) + '/' + str(model_prefix) + '/' + 'Keras_score_code.py', 'w')
    scorefile.write(parameters)
    scorefile.close()
    print('Score code generation complete')
    return None
