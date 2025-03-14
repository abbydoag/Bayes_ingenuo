import pandas as pd
import numpy as np
import random
#metrics
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.compose import make_column_selector as selector #separa
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB

train_df = pd.read_csv('train.csv')
pd.set_option('display.float_format', '{:.2f}'.format)
train_df.dropna(axis=1, how='all', inplace=True)

#reemplazando NaN (segun ejemplo)
num_cols = train_df.select_dtypes(include=['number']).columns
train_df[num_cols] = train_df[num_cols].apply(lambda col: col.fillna(col.median(numeric_only=True)))

y = train_df.pop('SalePrice')
X = train_df
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

nan_counts = X_train.isna().sum()

# Aplicar la mediana solo a columnas con más del 20% de NaN
threshold = 0.2 * len(X_train)
cols_to_impute = nan_counts[nan_counts > threshold].index

# Aplicar la mediana solo en esas columnas
X_train[cols_to_impute] = X_train[cols_to_impute].apply(lambda col: col.fillna(col.median()))
X_test[cols_to_impute] = X_test[cols_to_impute].apply(lambda col: col.fillna(col.median()))
cat_nan_counts = X_train.select_dtypes(include=['object']).isna().sum()
cat_cols_to_impute = cat_nan_counts[cat_nan_counts < 0.5 * len(X_train)].index
X_train[cat_cols_to_impute] = X_train[cat_cols_to_impute].apply(lambda col: col.fillna(col.mode()[0]))
X_test[cat_cols_to_impute] = X_test[cat_cols_to_impute].apply(lambda col: col.fillna(col.mode()[0]))

num_select = selector(dtype_exclude=object)
cat_select = selector(dtype_include=object)
num_col = num_select(train_df)
cat_col = cat_select(train_df)

cat_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
num_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', cat_preprocessor, cat_col),
    ('standar-scaler', num_preprocessor,num_col)
    ])


pipeline = Pipeline(
    [('preprocessor',preprocessor),
     ('regressor',GaussianNB())])
pipeline.get_params()

modelo = pipeline.fit(X_train,y_train) #ejecuta modelo que se creeo en el pipeline 
y_pred = modelo.predict(X_test) #da rvariable respuesta ya haceienod las predicciones
rmse = root_mean_squared_error(y_test,y_pred)
print(f"RMSE: {rmse}")