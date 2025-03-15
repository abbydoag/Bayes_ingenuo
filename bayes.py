import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#metrics
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.compose import make_column_selector as selector #separa
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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
cat_cols = X_train.select_dtypes(include=['object']).columns

num_imputer = SimpleImputer(strategy='median')
X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = num_imputer.transform(X_test[num_cols])
cat_imputer = SimpleImputer(strategy='most_frequent')
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
print(f"Bayes RMSE: {rmse}")
print(f"Bayes MAE: {mae}")
"""
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color="blue")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", linewidth=2)
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Predicciones vs. Valores Reales (SalePrice)")
plt.show()
"""


#-----------------------------------
# 8. validacion cruzada
#-----------------------------------
preprocessor_num = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
preprocessor_cat = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor_valC = ColumnTransformer([
    ('num', preprocessor_num, num_cols),
    ('cat', preprocessor_cat, cat_cols)
])

X_transformed = preprocessor_valC.fit_transform(X)

modelo = LinearRegression()
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

mae_score = cross_val_score(modelo, X_transformed, y, cv=kfold, scoring='neg_mean_absolute_error')
mae_prom = -mae_score.mean()
mse_score = cross_val_score(modelo, X_transformed, y, cv=kfold, scoring='neg_mean_squared_error')
mse_prom = -mse_score.mean()
rmse = np.sqrt(mse_prom)

print(f"Variacion Cruzada MAE: {mae_prom:.2f}")
print(f"Variacion Cruzada RMSE: {rmse:.2f}")