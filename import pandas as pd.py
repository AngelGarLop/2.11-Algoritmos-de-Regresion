import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, ElasticNet, BayesianRidge, Lasso, QuantileRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Cargar datos
student_data = pd.read_csv('StudentPerformanceFactors.csv')
housing_data = pd.read_csv('housing.csv', delim_whitespace=True, header=None)

# Preprocesamiento para StudentPerformanceFactors
X_student = student_data.drop('Exam_Score', axis=1)
y_student = student_data['Exam_Score']

# Preprocesamiento para housing
X_housing = housing_data.iloc[:, :-1]
y_housing = housing_data.iloc[:, -1]

# Función para preprocesar datos
def preprocess_data(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])
    return preprocessor

# Dividir datos
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluar métricas
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, r2, mae

# Función para entrenar y evaluar un modelo con diferentes hiperparámetros
def train_with_hyperparameters(model_class, hyperparams, X_train, X_test, y_train, y_test, preprocessor):
    results = {}
    for param_name, values in hyperparams.items():
        for value in values:
            # Crear modelo con el hiperparámetro ajustado
            model = model_class(**{param_name: value})
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('regressor', model)])
            pipeline.fit(X_train, y_train)
            # Evaluar modelo
            metrics = evaluate_model(pipeline, X_test, y_test)
            results[f"{param_name}={value}"] = metrics
    return results

# Procesar y dividir datos
preprocessor_student = preprocess_data(X_student)
X_train_student, X_test_student, y_train_student, y_test_student = split_data(X_student, y_student)

preprocessor_housing = preprocess_data(X_housing)
X_train_housing, X_test_housing, y_train_housing, y_test_housing = split_data(X_housing, y_housing)

# Hiperparámetros a probar
ridge_hyperparams = {'alpha': [0.1, 10.0], 'solver': ['auto', 'saga']}
elasticnet_hyperparams = {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.8]}
bayesian_hyperparams = {'alpha_1': [1e-6, 1e-4], 'alpha_2': [1e-6, 1e-4]}
lasso_hyperparams = {'alpha': [0.1, 1.0], 'max_iter': [1000, 5000]}
quantile_hyperparams = {'quantile': [0.25, 0.75], 'alpha': [0.1, 1.0]}

# Entrenar y evaluar modelos para StudentPerformanceFactors
print("Resultados para StudentPerformanceFactors:")

print("\nRidge:")
ridge_results_student = train_with_hyperparameters(Ridge, ridge_hyperparams, X_train_student, X_test_student, y_train_student, y_test_student, preprocessor_student)
for config, metrics in ridge_results_student.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nElasticNet:")
elasticnet_results_student = train_with_hyperparameters(ElasticNet, elasticnet_hyperparams, X_train_student, X_test_student, y_train_student, y_test_student, preprocessor_student)
for config, metrics in elasticnet_results_student.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nBayesian Ridge:")
bayesian_results_student = train_with_hyperparameters(BayesianRidge, bayesian_hyperparams, X_train_student, X_test_student, y_train_student, y_test_student, preprocessor_student)
for config, metrics in bayesian_results_student.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nLasso:")
lasso_results_student = train_with_hyperparameters(Lasso, lasso_hyperparams, X_train_student, X_test_student, y_train_student, y_test_student, preprocessor_student)
for config, metrics in lasso_results_student.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nQuantile Regression:")
quantile_results_student = train_with_hyperparameters(QuantileRegressor, quantile_hyperparams, X_train_student, X_test_student, y_train_student, y_test_student, preprocessor_student)
for config, metrics in quantile_results_student.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

# Entrenar y evaluar modelos para Housing
print("\nResultados para Housing:")

print("\nRidge:")
ridge_results_housing = train_with_hyperparameters(Ridge, ridge_hyperparams, X_train_housing, X_test_housing, y_train_housing, y_test_housing, preprocessor_housing)
for config, metrics in ridge_results_housing.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nElasticNet:")
elasticnet_results_housing = train_with_hyperparameters(ElasticNet, elasticnet_hyperparams, X_train_housing, X_test_housing, y_train_housing, y_test_housing, preprocessor_housing)
for config, metrics in elasticnet_results_housing.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nBayesian Ridge:")
bayesian_results_housing = train_with_hyperparameters(BayesianRidge, bayesian_hyperparams, X_train_housing, X_test_housing, y_train_housing, y_test_housing, preprocessor_housing)
for config, metrics in bayesian_results_housing.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nLasso:")
lasso_results_housing = train_with_hyperparameters(Lasso, lasso_hyperparams, X_train_housing, X_test_housing, y_train_housing, y_test_housing, preprocessor_housing)
for config, metrics in lasso_results_housing.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")

print("\nQuantile Regression:")
quantile_results_housing = train_with_hyperparameters(QuantileRegressor, quantile_hyperparams, X_train_housing, X_test_housing, y_train_housing, y_test_housing, preprocessor_housing)
for config, metrics in quantile_results_housing.items():
    print(f"{config} - RMSE: {metrics[0]}, R²: {metrics[1]}, MAE: {metrics[2]}")