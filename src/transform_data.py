import os
import typing
import pandas as pd
import numpy as np
import tsfel
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.decomposition
import sklearn.model_selection

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
  
  if os.path.exists(os.path.join(DATA_PATH, f'train_{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv')) and os.path.exists(os.path.join(DATA_PATH, f'test_{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv')):
    print('Skipping...', flush = True)
    continue
  
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

  X_train_combination.reset_index(drop = True).join(y_train).to_csv(os.path.join(DATA_PATH, f'train_{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv'), index = False)
  X_test_combination.reset_index(drop = True).join(y_test).to_csv(os.path.join(DATA_PATH, f'test_{str(combination["normalization"])}_{str(combination["logarithm"])}_{str(combination["differences"])}.csv'), index = False)
