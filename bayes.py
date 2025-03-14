import pandas as pd
import numpy as np
# Importar
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

# ------------------------------------------------------------------
# Inciso 4: Modelo de Clasificación usando la variable 'SalePrice_cat'
# ------------------------------------------------------------------
# Se vuelve a leer el archivo original para disponer de 'SalePrice', por si las moscas
df_class = pd.read_csv('train.csv')
df_class.dropna(axis=1, how='all', inplace=True)

# Crear la variable categórica 'SalePrice_cat' (con etiquetas 'barata', 'media' y 'cara')
bins = [df_class['SalePrice'].min(), 
        df_class['SalePrice'].quantile(0.33), 
        df_class['SalePrice'].quantile(0.66), 
        df_class['SalePrice'].max()]
labels_cat = ['barata', 'media', 'cara']
df_class['SalePrice_cat'] = pd.cut(df_class['SalePrice'], bins=bins, labels=labels_cat, include_lowest=True)

# Preparar los datos para clasificación: se elimina 'SalePrice' y se utiliza 'SalePrice_cat' como respuesta
X_class = df_class.drop(['SalePrice', 'SalePrice_cat'], axis=1)
y_cat = df_class['SalePrice_cat']

# Realizar imputación en X_class para evitar NaN por el amor a DIOS
num_cols_class = X_class.select_dtypes(include=['number']).columns
cat_cols_class = X_class.select_dtypes(include=['object']).columns

num_imputer_class = SimpleImputer(strategy='median')
cat_imputer_class = SimpleImputer(strategy='most_frequent')

X_class[num_cols_class] = num_imputer_class.fit_transform(X_class[num_cols_class])
X_class[cat_cols_class] = cat_imputer_class.fit_transform(X_class[cat_cols_class])

# Dividir datos para clasificación (70% entrenamiento, 30% prueba)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_class, y_cat, train_size=0.7, test_size=0.3, random_state=42)

# Preprocesamiento para clasificación: escalado para numéricas y one-hot encoding para categóricas
num_features_clf = selector(dtype_exclude=object)(X_class)
cat_features_clf = selector(dtype_include=object)(X_class)

preprocessor_clf = ColumnTransformer([
    ('num', StandardScaler(), num_features_clf),
    ('cat', OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features_clf)
])

# Crear pipeline de clasificación usando GaussianNB
pipeline_nb_clf = Pipeline([
    ('preprocessor', preprocessor_clf),
    ('classifier', GaussianNB())
])
pipeline_nb_clf.fit(X_train_clf, y_train_clf)
y_pred_clf = pipeline_nb_clf.predict(X_test_clf)

# ------------------------------------------------------------------
# Inciso 5: Evaluación del modelo de clasificación
# ------------------------------------------------------------------
# Calcular accuracy, reporte de clasificación y validación cruzada
accuracy_clf = accuracy_score(y_test_clf, y_pred_clf)
print("\nClasificación - Accuracy (Inciso 5):", accuracy_clf)
print("\nReporte de Clasificación (Inciso 5):")
print(classification_report(y_test_clf, y_pred_clf))

cv_scores_clf = cross_val_score(pipeline_nb_clf, X_train_clf, y_train_clf, cv=5, scoring='accuracy')
print("\nCV Accuracy scores (GaussianNB) (Inciso 5):", cv_scores_clf)
print("Mean CV Accuracy (Inciso 5):", np.mean(cv_scores_clf))

# ------------------------------------------------------------------
# Inciso 6: Análisis de la eficiencia del modelo de clasificación mediante Matriz de Confusión
# ------------------------------------------------------------------
cm = confusion_matrix(y_test_clf, y_pred_clf)
print("\nMatriz de Confusión (Inciso 6):")
print(cm)



#-----------------------------------
# 8. validacion cruzada
#-----------------------------------

