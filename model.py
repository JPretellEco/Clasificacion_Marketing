import os
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ------------------------------
# RUTAS DE ARCHIVOS
# ------------------------------

session_test = 'C:/Users/leo_2/Documents/desafio/nuwe-data-ds1/test/session_test.csv'
user_test = 'C:/Users/leo_2/Documents/desafio/nuwe-data-ds1/test/user_test.csv'

session_train = 'C:/Users/leo_2/Documents/desafio/nuwe-data-ds1/train/session_train.csv'
user_train = 'C:/Users/leo_2/Documents/desafio/nuwe-data-ds1/train/user_train.csv'

# ------------------------------
# FUNCIONES DE PREPROCESAMIENTO
# ------------------------------

def build_features(df_users, df_sessions):
    agg = df_sessions.groupby('user_id').agg({
        'session_id': 'count',
        'page_views': ['sum', 'mean'],
        'session_duration': ['sum', 'mean', 'max'],
        'device_type': pd.Series.nunique,
        'browser': pd.Series.nunique,
        'operating_system': pd.Series.nunique,
        'country': pd.Series.nunique,
        'search_query': 'count'
    })
    agg.columns = ['_'.join(col).strip() for col in agg.columns.values]
    agg = agg.reset_index()

    df_users = df_users.copy()
    df_users['abandoned_cart'] = df_users['abandoned_cart'].astype(int)
    df_users['user_category'] = df_users['user_category'].map({
        'new_user': 0,
        'recurring_user': 1,
        'premium_subscriber': 2
    })

    return df_users.merge(agg, on='user_id', how='left')


def asignar_generacion(anio_nacimiento):
    if anio_nacimiento >= 1998:
        return 'Gen Z'
    elif anio_nacimiento >= 1982:
        return 'Millennial'
    elif anio_nacimiento >= 1966:
        return 'Gen X'
    else:
        return 'Baby Boomer'


def transformar_generacion(df):
    df['birth_year'] = 2025 - df['age']
    df['generation'] = df['birth_year'].apply(asignar_generacion)
    df.drop(columns=['birth_year', 'age'], inplace=True)
    df['generation'] = df['generation'].map({
        'Gen Z': 0,
        'Millennial': 1,
        'Gen X': 2,
        'Baby Boomer': 3
    })
    return df


def preparar_datos(ruta_user, ruta_session, is_train=True):
    user_df = pd.read_csv(ruta_user, sep=';')
    session_df = pd.read_csv(ruta_session)
    features = build_features(user_df, session_df)
    features = transformar_generacion(features)
    if is_train:
        X = features.drop(columns=['user_id', 'marketing_target'])
        y = features['marketing_target']
        return X, y
    else:
        ids = user_df[['user_id', 'test_id']]
        X = features.drop(columns=['user_id', 'test_id'])
        return X, ids


def entrenar_modelo(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_val_scaled)

    print("🔍 Evaluación del modelo:")
    print(f"Accuracy:  {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"Recall:    {recall_score(y_val, y_pred, average='weighted'):.4f}")
    print(f"F1 Score:  {f1_score(y_val, y_pred, average='weighted'):.4f}")

    return model, scaler


def predecir_y_exportar(model, scaler, X_test, ids_test, output_path='C:/Users/leo_2/Documents/desafio/nuwe-data-ds1/predictions/predicciones.json'):
    # Escalar X_test
    X_test_scaled = scaler.transform(X_test)
    preds = model.predict(X_test_scaled)

    # Crear DataFrame con test_id y predicción
    results_df = pd.DataFrame({
        'test_id': ids_test['test_id'].values,
        'pred': preds
    })

    # Ordenar por test_id (convertido a int por seguridad)
    results_df['test_id'] = results_df['test_id'].astype(int)
    results_df.sort_values(by='test_id', inplace=True)

    # Convertir a diccionario {test_id: pred}
    results_dict = {
        "target": {str(row['test_id']): int(row['pred']) for _, row in results_df.iterrows()}
    }

    # Exportar a JSON
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    print(f"✅ Predicciones exportadas en orden a {output_path}")




# --------------------------
# EJECUCIÓN DEL PIPELINE
# --------------------------
if __name__ == "__main__":
    print("📦 Preparando datos de entrenamiento...")
    X_train, y_train = preparar_datos(user_train, session_train, is_train=True)

    print("🧠 Entrenando modelo...")
    modelo, scaler = entrenar_modelo(X_train, y_train)

    print("🧪 Cargando datos de prueba...")
    X_test, test_ids = preparar_datos(user_test, session_test, is_train=False)

    print("📤 Generando predicciones finales...")
    predecir_y_exportar(modelo, scaler, X_test, test_ids)
