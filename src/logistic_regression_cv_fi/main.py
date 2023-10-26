import os
import pandas as pd
import numpy as np
import sklearn.model_selection
import sklearn.metrics
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

DATA_PATH = '../../data'

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
]

CHUNK = 10

STEP = 566
FROM =  STEP * (CHUNK - 1) 
TO = STEP * CHUNK if CHUNK < 10 else 5657 # Total rows

# ===========================================================================
# ============================= Core Experiments ============================
# ===========================================================================

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

  predictions = pd.DataFrame()
  feature_importance = pd.DataFrame()
  
  # Grid search over parameters
  for parameter_set in sklearn.model_selection.ParameterGrid(model['parameters']):
    print('>' * 10, parameter_set, '<' * 10, flush = True)

    for test_index in range(FROM, TO):
      train_indices = list(range(0, 5657))
      train_indices.remove(test_index)
      test_indices = [test_index]

      X_train, X_test = X_combination.iloc[train_indices], X_combination.iloc[test_indices]
      y_train, y_test = y_combination[train_indices], y_combination[test_indices]

      classifier = model['executable']()
      classifier.set_params(**parameter_set)
      classifier.fit(X_train, y_train)

      predictions = pd.concat([
        predictions,
        pd.DataFrame({
          'real' : y_test.values,
          'prediction' : classifier.predict(X_test),
        })
      ])

      # Feature importance using coefficients
      coefficient_weights = pd.DataFrame([i for i in zip(classifier.feature_names_in_, classifier.coef_[0])], columns = ['feature', 'coefficient'])
      coefficient_weights['absolute_coefficient'] = coefficient_weights['coefficient'].apply(lambda x : np.abs(x))
      coefficient_weights['absolute_rank'] = coefficient_weights['absolute_coefficient'].rank(ascending = False)
      feature_importance = pd.concat([
        feature_importance,
        coefficient_weights
      ])

  if not os.path.exists(os.path.join(DATA_PATH, 'feature_importance', f'{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}')):
    os.makedirs(os.path.join(DATA_PATH, 'feature_importance', f'{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}'))

  if not os.path.exists(os.path.join(DATA_PATH, 'cross_validation', f'{model["name"]}_{parameter_set}{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}')):
    os.makedirs(os.path.join(DATA_PATH, 'cross_validation', f'{model["name"]}_{parameter_set}{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}'))

  feature_importance.reset_index(drop = True).to_csv(os.path.join(DATA_PATH, 'feature_importance', f'{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}', f'{FROM}_{TO}.csv'), index = False)
  predictions.reset_index(drop = True).to_csv(os.path.join(DATA_PATH, 'cross_validation', f'{model["name"]}_{parameter_set}{str(model["normalization"])}_{str(model["logarithm"])}_{str(model["differences"])}', f'{FROM}_{TO}.csv'), index = False)
