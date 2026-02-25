import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=420)

# Average CV score on the training set was: -0.0026707796914211546
exported_pipeline = XGBRegressor(learning_rate=0.001, max_depth=4, min_child_weight=11, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.25, verbosity=0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 420)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
