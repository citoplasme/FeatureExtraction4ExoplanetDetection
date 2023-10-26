import os
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#import warnings
#warnings.filterwarnings('ignore')

# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

DATA_PATH = '../data'

MODELS = [
  # {
  #   'normalization' : True,
  #   'logarithm' : True,
  #   'differences' : True,
  #   'name' : 'Logistic Regression',
  #   'executable' : LogisticRegression,
  #   'parameters' : {
  #     'max_iter' : [100000],
  #     'penalty' : ['l1'],
  #     'C' : [0.1],
  #     'solver' : ['saga'],
  #     'n_jobs' : [-1],
  #   }
  # },
  {
    'normalization' : False,
    'logarithm' : True,
    'differences' : True,
    'name' : 'Logistic Regression',
    'executable' : LogisticRegression,
    'parameters' : {
      'max_iter' : [100000],
      'penalty' : ['l1'],
      'C' : [0.1],
      'solver' : ['saga'],
      'n_jobs' : [-1],
    }
  },
  # {
  #   'normalization' : False,
  #   'logarithm' : False,
  #   'differences' : True,
  #   'name' : 'Logistic Regression',
  #   'executable' : LogisticRegression,
  #   'parameters' : {
  #     'max_iter' : [100000],
  #     'penalty' : ['l2'],
  #     'C' : [1],
  #     'solver' : ['saga'],
  #     'n_jobs' : [-1],
  #   }
  # },
  # {
  #   'normalization' : False,
  #   'logarithm' : True,
  #   'differences' : True,
  #   'name' : 'Random Forest',
  #   'executable' : RandomForestClassifier,
  #   'parameters' : {
  #     'criterion' : ['gini'],
  #     'max_depth' : [10],
  #     'min_samples_split' : [5],
  #     'n_estimators' : [10],
  #     'n_jobs' : [-1],
  #   }
  # },
  # {
  #   'normalization' : True,
  #   'logarithm' : True,
  #   'differences' : True,
  #   'name' : 'K-Nearest Neighbours',
  #   'executable' : KNeighborsClassifier,
  #   'parameters' : {
  #     'algorithm' : ['ball_tree'],
  #     'n_neighbors' : [5],
  #     'p' : [2],
  #     'weights' : ['distance'],
  #     'n_jobs' : [-1],
  #   }
  # },
  # {
  #   'normalization' : False,
  #   'logarithm' : True,
  #   'differences' : True,
  #   'name' : 'Support Vector Machine',
  #   'executable' : SVC,
  #   'parameters' : {
  #     'C' : [1],
  #     'gamma' : ['scale'],
  #     'kernel' : ['sigmoid'],
  #   }
  # },
  # {
  #   'normalization' : True,
  #   'logarithm' : True,
  #   'differences' : True,
  #   'name' : 'Support Vector Machine',
  #   'executable' : SVC,
  #   'parameters' : {
  #     'C' : [10],
  #     'gamma' : [1],
  #     'kernel' : ['poly'],
  #   }
  # },
  # {
  #   'normalization' : False,
  #   'logarithm' : False,
  #   'differences' : True,
  #   'name' : 'Support Vector Machine',
  #   'executable' : SVC,
  #   'parameters' : {
  #     'C' : [10],
  #     'gamma' : [1],
  #     'kernel' : ['poly'],
  #   }
  # },
]

LEAVE_ONE_OUT = sklearn.model_selection.LeaveOneOut()

# ===========================================================================
# ============================= Core Experiments ============================
# ===========================================================================

performance_metrics = pd.DataFrame(columns = ['class', 'normalization', 'logarithm', 'differences', 'algorithm', 'hyper_parameters', 'precision', 'recall', 'f1_score'])

for model in MODELS:
  print('#' * 20, model['name'].center(53), '#' * 20, flush = True)
  
  training_set = pd.read_csv(os.path.join(DATA_PATH, f'train_{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}.csv'))
  testing_set = pd.read_csv(os.path.join(DATA_PATH, f'test_{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}.csv'))

  complete_data_set = pd.concat([training_set, testing_set]).reset_index(drop = True)
  X_combination = complete_data_set.select_dtypes('number')
  y_combination = complete_data_set['exoplanets']

  # ===========================================================================
  # ====================== Leave-one-out Cross-validation =====================
  # ===========================================================================

  predictions = list()
  # Grid search over parameters
  for parameter_set in sklearn.model_selection.ParameterGrid(model['parameters']):
    print('>' * 10, parameter_set, '<' * 10, flush = True)

    for train_indices, test_indices in LEAVE_ONE_OUT.split(X_combination):
      X_train, X_test = X_combination.iloc[train_indices], X_combination.iloc[test_indices]
      y_train, y_test = y_combination[train_indices], y_combination[test_indices]
              
      classifier = model['executable']()
      classifier.set_params(**parameter_set)
      classifier.fit(X_train, y_train)

      predictions.append(classifier.predict(X_test)[0])

  performance_report = sklearn.metrics.classification_report(y_combination, predictions, target_names = ['NO_EXOPLANETS', 'EXOPLANETS'], output_dict = True)

  performance_metrics = pd.concat([
    performance_metrics,
    pd.DataFrame( {
      'class' : ['NO_EXOPLANETS', 'EXOPLANETS'],
      'normalization' : [int(model['normalization']), int(model['normalization'])],
      'logarithm' : [int(model['logarithm']), int(model['logarithm'])],
      'differences' : [int(model['differences']), int(model['differences'])],
      'algorithm' : [model['name'], model['name']],
      'hyper_parameters' : [parameter_set, parameter_set],
      'accuracy' : [np.around(performance_report['accuracy'], decimals = 5), np.around(performance_report['accuracy'], decimals = 5)],
      'precision' : [np.around(performance_report['NO_EXOPLANETS']['precision'], decimals = 5), np.around(performance_report['EXOPLANETS']['precision'], decimals = 5)],
      'recall' : [np.around(performance_report['NO_EXOPLANETS']['recall'], decimals = 5), np.around(performance_report['EXOPLANETS']['recall'], decimals = 5)],
      'f1_score' : [np.around(performance_report['NO_EXOPLANETS']['f1-score'], decimals = 5), np.around(performance_report['EXOPLANETS']['f1-score'], decimals = 5)],
    }),
  ])

  performance_metrics.reset_index(drop = True).to_csv(os.path.join(DATA_PATH, 'cross_validation', f'{model["name"]}_{parameter_set}{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}.csv'), index = False)
