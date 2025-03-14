import pandas as pd
import numpy as np
#metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import make_column_selector as selector #separa
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

train_df = pd.read_csv('train.csv')
test_def = pd.read_csv('test.csv')
pd.set_option('display.float_format', '{:.2f}'.format)
train_df.dropna(axis=1, how='all', inplace=True)

#-----------------------------------
# 1. Bayes Modelo
#-----------------------------------
y = train_df.pop('SalePrice')
X = train_df
y_labels = pd.cut(y, bins=3, labels=['Bajo', 'Medio', 'Alto'])
X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.3,train_size=0.7)

num_cols = X_train.select_dtypes(include=['number']).columns
num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])

# Aplicar la mediana solo en esas columnas
cat_cols = X_train.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')  # Impute with the most frequent value (mode)
X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])

cat_preprocessor = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
num_preprocessor = StandardScaler()

preprocessor = ColumnTransformer([
    ('one-hot-encoder', cat_preprocessor, cat_cols),
    ('standar-scaler', num_preprocessor,num_cols)
    ])
pipeline = Pipeline(
    [('preprocessor',preprocessor),
     ('regressor', BayesianRidge())
    ])

bayes = pipeline.fit(X_train,y_train) #ejecuta modelo que se creeo en el pipeline 
y_pred = bayes.predict(X_test) #da rvariable respuesta ya haceienod las predicciones
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")




#-----------------------------------
# 8. validacion cruzada
#-----------------------------------

