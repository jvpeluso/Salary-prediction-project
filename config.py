import numpy as np

# File paths
TRAIN_FEATURES= 'myData/train_features.csv'
TRAIN_TARGET = 'myData/train_target.csv'
TEST_FEATURES = 'myData/test_features.csv'
TEST_TARGET = 'myData/test_target.csv'

# Features and target names
FEATURES = ['jobId','companyId','jobType','degree','major','industry', 'yearsExperience','milesFromMetropolis']
TARGET = 'salary'
TARGET_PRETTY = 'Salary'
NUM_FEATURES = ['yearsExperience','milesFromMetropolis']
NUM_FEATURES_PRETTY = ['Years of Experience','Miles from Metropolis']
NUM_FEATURES_TARGET = ['yearsExperience','milesFromMetropolis','salary']
NUM_FEATURES_TARGET_PRETTY = ['Years of Experience','Miles from Metropolis','Salary']
CAT_FEATURES = ['jobId','companyId','jobType','degree','major','industry']
CAT_FEATURES_PRETTY = ['Job ID','Company Id','Job Type','Degree','Major','Industry']
LOG_FEATURES = ['mean_SalaryGroup', 'std_SalaryGroup', 'min_SalaryGroup', 'max_SalaryGroup']
FINAL_FEATURES = ['companyId', 'jobType', 'degree', 'major', 'industry', 'yearsExperience', 'milesFromMetropolis',
                  'hasMajor', 'mean_SalaryGroup', 'std_SalaryGroup', 'min_SalaryGroup', 'max_SalaryGroup']

# Default global configurations
RANDOM_SEED = 14

# Default configurations for data processing
MERGE_COL = 'jobId'
PRINT_INFO = True
CHECK_ZEROS_TRAIN = True
DEL_NAN_TRAIN = True
DEL_DUPL_TRAIN = True
CHECK_ZEROS_TEST = False
DEL_NAN_TEST = True
DEL_DUPL_TEST = False
DEL_DUPL_COLS = 'jobId'

# Default configurations for plots

PLOT_COLORS = ['b','g','r','m','y', 'silver', 'cadetblue', 'yellowgreen', 'darksalmon'] 

# Default configurations for data engineering
DUMMY_FEAT = 'major'
NEW_DUMMY_FEAT = 'hasMajor'
DUMMY_LAMBDA = ['NONE', 0, 1]
NEW_FEATURES_GROUP = ['jobType', 'industry', 'degree', 'major', 'yearsExperience']
NEW_FEATURES = ['mean_SalaryGroup', 'std_SalaryGroup', 'min_SalaryGroup', 'max_SalaryGroup']
NEW_FEATURES_METHOD = {'mean' : np.mean, 'std_' : np.std, 'min_' : np.min, 'max_' : np.max}
DCT_NEW_PATH = ['myData/meanDct.json', 'myData/stdDct.json', 'myData/minDct.json', 'myData/maxDct.json']
CAT_FEATURES_ENCODE = ['jobType','degree','major','industry']
DCT_ENC_PATH = ['myData/jobDct.json', 'myData/degDct.json', 'myData/majDct.json', 'myData/indDct.json']
TRAIN_VAL_PERC = 0.2
DROP_FEATURES = 'jobId'
COMP_FEATURE = 'companyId'

# Default configurations for model development
BASELINE_FEATURES = ['jobType', 'companyId']
MODEL_PATH = 'myData/bestModel.pkl'
PREDICTION_PATH = 'myData/predictions.csv'

