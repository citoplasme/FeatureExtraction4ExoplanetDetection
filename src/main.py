import os
import typing
import pandas as pd
import numpy as np
import tsfel
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.model_selection
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# ============================ Global Parameters ============================
# ===========================================================================

DATA_PATH = '../data'

NORMALIZATION = [False, True]
LOGARITHM = [False, True]
DIFFERENCES = [False, True]

COMBINATIONS = pd.DataFrame(
  [(n, l, d) for n in NORMALIZATION for l in LOGARITHM for d in DIFFERENCES],
  columns = ['normalization', 'logarithm', 'differences']
)

SCALER = sklearn.preprocessing.MinMaxScaler(feature_range = (0,1))

SAMPLING_FREQUENCY = round(1 / ((80 * 24 * 60 * 60) / 3197), 6)

FEATURE_CONFIG = tsfel.get_features_by_domain()
FEATURE_CONFIG.get('statistical').get('Median absolute deviation').update({'use' : 'no'})

CORRELATION_THRESHOLD = 0.95

STANDARD_SCALER = sklearn.preprocessing.StandardScaler()

ALGORITHMS = [
  {
    'name' : 'Logistic Regression',
    'executable' : LogisticRegression,
    'parameters' : {
      'max_iter' : [100000],
      'penalty' : [None, 'l2'],
      'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
      'solver' : ['lbfgs', 'sag'],
      'n_jobs' : [-1],
    }
  },
  {
    'name' : 'Logistic Regression',
    'executable' : LogisticRegression,
    'parameters' : {
      'max_iter' : [100000],
      'penalty' : [None, 'l1', 'l2'],
      'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
      'solver' : ['saga'],
      'n_jobs' : [-1],
    }
  },
  {
    'name' : 'Logistic Regression',
    'executable' : LogisticRegression,
    'parameters' : {
      'max_iter' : [100000],
      'penalty' : ['elasticnet'],
      'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
      'solver' : ['saga'],
      'l1_ratio' : [0, 0.5, 1],
      'n_jobs' : [-1],
    }
  },
  {
    'name' : 'K-Nearest Neighbours',
    'executable' : KNeighborsClassifier,
    'parameters' : {
      'n_neighbors' : [2, 3, 5, 8, 10],
      'weights' : ['uniform', 'distance'],
      'p' : [1, 2],
      'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
      'n_jobs' : [-1],
    }
  },
  {
    'name' : 'Random Forest',
    'executable' : RandomForestClassifier,
    'parameters' : {
      'n_estimators' : [10, 100, 200],
      'criterion' : ['gini', 'entropy'],
      'max_depth' : [None, 10, 20],
      'min_samples_split' : [2, 5, 10],
      #'min_samples_leaf' : [1, 2, 4],
      #'max_features' : [None, 'sqrt', 'log2'],
      'n_jobs' : [-1],
    }
  },
  {
    'name' : 'Support Vector Machine',
    'executable' : SVC,
    'parameters' : {
      'C' : [0.001, 0.01, 0.1, 1, 10],
      #'degree' : [2, 3, 4, 5],
      'gamma' : [1, 'scale', 'auto'],
      'kernel' : ['poly', 'linear', 'rbf', 'sigmoid'],
    }
  },
]

VARIANCE_THRESOLDS = [0.99, 0.95, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]

# ===========================================================================
# =========================== Auxiliary Functions ===========================
# ===========================================================================

def load_data(path : str) -> pd.DataFrame:
  df = pd.read_csv(path)
  df['LABEL'] = (df['LABEL'] - 1).astype(bool)
  df = df.rename(columns = {'LABEL' : 'exoplanets'}).rename(lambda x : x.lower().replace('flux.', 't-'), axis = 'columns')
  df = df.sample(frac = 1).reset_index(drop = True)
  return df

def normalize(df : pd.DataFrame) -> pd.DataFrame:
  return pd.DataFrame(
    SCALER.fit_transform(df.T).T,
    index = df.index, columns = df.columns
  )

def logarithm(df : pd.DataFrame) -> pd.DataFrame:
  return df.apply(lambda x : np.sign(x) * np.log(np.abs(x) + 1), raw = True)

def differences(df : pd.DataFrame) -> pd.DataFrame:
  return df.diff(axis = 1).drop(columns = ['t-1'])

def extract_features(df : pd.DataFrame) -> pd.DataFrame:
  df_features = pd.DataFrame()
  for _, row in df.iterrows():
    df_features = pd.concat([
      df_features,
      tsfel.time_series_features_extractor(FEATURE_CONFIG, row, verbose = False, fs = SAMPLING_FREQUENCY)
    ])
  return df_features

def drop_features_correlation(df : pd.DataFrame) -> typing.List[str]:
  correlation_matrix = np.absolute(df.corr())
  correlation_matrix_upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k = 1).astype(bool))
  return [column for column in correlation_matrix_upper.columns if any(correlation_matrix_upper[column] > CORRELATION_THRESHOLD)]

def drop_features_variance(df : pd.DataFrame, variance_selector : sklearn.feature_selection.VarianceThreshold) -> pd.DataFrame:
  column_indices = variance_selector.get_support(indices = True)
  return df[df.columns[column_indices]]

def scale_features(df : pd.DataFrame, fit : bool = False) -> pd.DataFrame:
  if fit:
    return pd.DataFrame(
      STANDARD_SCALER.fit_transform(df),
      index = df.index, columns = df.columns
    )
  else:
    return pd.DataFrame(
      STANDARD_SCALER.transform(df),
      index = df.index, columns = df.columns
    )
  
# ===========================================================================
# ============================= Core Experiments ============================
# ===========================================================================

training_set = load_data(os.path.join(DATA_PATH, 'train.csv'))
testing_set = load_data(os.path.join(DATA_PATH, 'test.csv'))

X_train = training_set.select_dtypes('number')
y_train = training_set['exoplanets']

X_test = testing_set.select_dtypes('number')
y_test = testing_set['exoplanets']


for _, combination in COMBINATIONS.iterrows():
  print('-' * 20, 'Normalization:', combination['normalization'], 'Logarithm:', combination['logarithm'], 'Differences:', combination['differences'], '-' * 20, flush = True)
  
  if os.path.exists(os.path.join(DATA_PATH, 'performance_metrics', f'{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv')):
    print('Skipping...', flush = True)
    continue

  performance_metrics = pd.DataFrame(columns = ['class', 'normalization', 'logarithm', 'differences', 'algorithm', 'hyper_parameters', 'precision', 'recall', 'f1_score', 'method', 'variance_threshold'])
  
  X_train_combination = X_train.copy()
  X_test_combination = X_test.copy()
  
  # Row-wise normalization
  if combination['normalization']:
    X_train_combination = normalize(X_train_combination)
    X_test_combination = normalize(X_test_combination)
  
  # Logarithm to deal with heteroscedasticity
  if combination['logarithm']:
    X_train_combination = logarithm(X_train_combination)
    X_test_combination = logarithm(X_test_combination)
  
  # First differences to deal with trend
  if combination['differences']:
    X_train_combination = differences(X_train_combination)
    X_test_combination = differences(X_test_combination)

  # ===========================================================================
  # ========================= Feature-based Experiments =======================
  # ===========================================================================

  print('*' * 20, 'Feature-based Experiments'.center(53), '*' * 20, flush = True)

  # Feature extraction
  X_train_combination = extract_features(X_train_combination)
  X_test_combination = extract_features(X_test_combination)

  # Drop features based on correlation
  features_to_drop_correlation = drop_features_correlation(X_train_combination)
  X_train_combination.drop(features_to_drop_correlation, axis = 1, inplace = True)
  X_test_combination.drop(features_to_drop_correlation, axis = 1, inplace = True)

  # Drop features based on having zero variance
  variance_selector = sklearn.feature_selection.VarianceThreshold()
  variance_selector.fit(X_train_combination)
  X_train_combination = drop_features_variance(X_train_combination, variance_selector)
  X_test_combination = drop_features_variance(X_test_combination, variance_selector)

  # Scale features
  X_train_combination = scale_features(X_train_combination, fit = True)
  X_test_combination = scale_features(X_test_combination, fit = False)

  for algorithm in ALGORITHMS:
    print('#' * 20, algorithm['name'].center(53), '#' * 20, flush = True)
    classifier = algorithm['executable']()
    # Grid search over parameters
    for parameter_set in sklearn.model_selection.ParameterGrid(algorithm['parameters']):
      print('>' * 10, parameter_set, '<' * 10, flush = True)
      classifier.set_params(**parameter_set)

      classifier.fit(X_train_combination, y_train)

      predictions = classifier.predict(X_test_combination)

      #print(sklearn.metrics.classification_report(y_test, predictions, target_names = ['NO_EXOPLANETS', 'EXOPLANETS'], digits = 5), flush = True)
      #print(sklearn.metrics.confusion_matrix(y_test, predictions), flush = True)

      performance_report = sklearn.metrics.classification_report(y_test, predictions, target_names = ['NO_EXOPLANETS', 'EXOPLANETS'], output_dict = True)

      performance_metrics = pd.concat([
        performance_metrics,
        pd.DataFrame( {
          'class' : ['NO_EXOPLANETS', 'EXOPLANETS'],
          'normalization' : [int(combination['normalization']), int(combination['normalization'])],
          'logarithm' : [int(combination['logarithm']), int(combination['logarithm'])],
          'differences' : [int(combination['differences']), int(combination['differences'])],
          'algorithm' : [algorithm['name'], algorithm['name']],
          'hyper_parameters' : [parameter_set, parameter_set],
          'accuracy' : [np.around(performance_report['accuracy'], decimals = 5), np.around(performance_report['accuracy'], decimals = 5)],
          'precision' : [np.around(performance_report['NO_EXOPLANETS']['precision'], decimals = 5), np.around(performance_report['EXOPLANETS']['precision'], decimals = 5)],
          'recall' : [np.around(performance_report['NO_EXOPLANETS']['recall'], decimals = 5), np.around(performance_report['EXOPLANETS']['recall'], decimals = 5)],
          'f1_score' : [np.around(performance_report['NO_EXOPLANETS']['f1-score'], decimals = 5), np.around(performance_report['EXOPLANETS']['f1-score'], decimals = 5)],
          'method' : ['features', 'features'],
          'variance_threshold' : [None, None]
        }),
      ])
  
  # ===========================================================================
  # =========================== PCA-based Experiments =========================
  # ===========================================================================

  print('*' * 20, 'PCA-based Experiments'.center(53), '*' * 20, flush = True)

  for variance_threshold in VARIANCE_THRESOLDS:
    pca = sklearn.decomposition.PCA(variance_threshold)
    
    X_train_combination_pca = pca.fit_transform(X_train_combination)
    X_test_combination_pca = pca.transform(X_test_combination)

    X_train_combination_pca = pd.DataFrame(X_train_combination_pca, columns = ['principal_component_' + str(i) for i in range(1, X_train_combination_pca.shape[1] + 1)], index = X_train_combination.index)
    X_test_combination_pca = pd.DataFrame(X_test_combination_pca, columns = ['principal_component_' + str(i) for i in range(1, X_test_combination_pca.shape[1] + 1)], index = X_test_combination.index)
    
    for algorithm in ALGORITHMS:
      print('#' * 20, algorithm['name'].center(53), '#' * 20, flush = True)
      classifier = algorithm['executable']()
      # Grid search over parameters
      for parameter_set in sklearn.model_selection.ParameterGrid(algorithm['parameters']):
        print('>' * 10, parameter_set, '<' * 10, flush = True)
        classifier.set_params(**parameter_set)

        classifier.fit(X_train_combination, y_train)

        predictions = classifier.predict(X_test_combination)

        #print(sklearn.metrics.classification_report(y_test, predictions, target_names = ['NO_EXOPLANETS', 'EXOPLANETS'], digits = 5), flush = True)
        #print(sklearn.metrics.confusion_matrix(y_test, predictions), flush = True)

        performance_report = sklearn.metrics.classification_report(y_test, predictions, target_names = ['NO_EXOPLANETS', 'EXOPLANETS'], output_dict = True)

        performance_metrics = pd.concat([
          performance_metrics,
          pd.DataFrame( {
            'class' : ['NO_EXOPLANETS', 'EXOPLANETS'],
            'normalization' : [int(combination['normalization']), int(combination['normalization'])],
            'logarithm' : [int(combination['logarithm']), int(combination['logarithm'])],
            'differences' : [int(combination['differences']), int(combination['differences'])],
            'algorithm' : [algorithm['name'], algorithm['name']],
            'hyper_parameters' : [parameter_set, parameter_set],
            'accuracy' : [np.around(performance_report['accuracy'], decimals = 5), np.around(performance_report['accuracy'], decimals = 5)],
            'precision' : [np.around(performance_report['NO_EXOPLANETS']['precision'], decimals = 5), np.around(performance_report['EXOPLANETS']['precision'], decimals = 5)],
            'recall' : [np.around(performance_report['NO_EXOPLANETS']['recall'], decimals = 5), np.around(performance_report['EXOPLANETS']['recall'], decimals = 5)],
            'f1_score' : [np.around(performance_report['NO_EXOPLANETS']['f1-score'], decimals = 5), np.around(performance_report['EXOPLANETS']['f1-score'], decimals = 5)],
            'method' : ['principal_components', 'principal_components'],
            'variance_threshold' : [variance_threshold, variance_threshold]
          }),
        ])

  print('', flush = True)

  performance_metrics.reset_index(drop = True).to_csv(os.path.join(DATA_PATH, 'performance_metrics', f'{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv'), index = False)