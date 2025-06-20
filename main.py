import os
import pandas as pd
import spacy
import joblib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from spacy.lang.es.stop_words import STOP_WORDS as SPANISH_STOP_WORDS
from normalizacion.normalizacion import normalizar

# Configuración
CORPUS_PATH = os.path.join('dataset', 'Dataset_fakenews.csv')
OUTPUT_FOLDER = 'corpus/normalizacion_mejorada'
MODEL_PATH = os.path.join('modelos', 'xgb_optimizado.pkl')

def cargar_datos():
    """Carga el corpus original"""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"No se encontró el archivo en {CORPUS_PATH}")
    return pd.read_csv(CORPUS_PATH, encoding='utf-8')

def aplicar_normalizacion(datos):
    """Aplica la normalización mejorada"""
    nlp = spacy.load('es_core_news_sm')
    datos_normalizados = datos.copy()
    datos_normalizados['Headline'] = datos_normalizados['Headline'].apply(
        lambda x: normalizar(str(x), nlp))
    return datos_normalizados

def agregar_caracteristicas(df):
    """Agrega características avanzadas"""
    # Características básicas
    df['longitud'] = df['Headline'].apply(len)
    df['mayusculas'] = df['Headline'].apply(lambda x: sum(1 for c in x if c.isupper()))
    df['exclamaciones'] = df['Headline'].str.count(r'[¡!]')
    df['interrogaciones'] = df['Headline'].str.count(r'[¿?]')
    
    # Nuevas características
    df['prop_mayusculas'] = df['mayusculas'] / df['longitud']
    df['densidad_puntuacion'] = (df['exclamaciones'] + df['interrogaciones']) / df['longitud']
    
    # Palabras sensacionalistas
    sensacionalistas = ['descubren', 'impactante', 'escándalo', 'polémica', 'revelan']
    for word in sensacionalistas:
        df[f'sens_{word}'] = df['Headline'].str.contains(word, case=False).astype(int)
    
    return df

def crear_pipeline():
    """Pipeline con XGBoost y características avanzadas"""
    # Procesamiento de texto
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=15000,
            stop_words=list(SPANISH_STOP_WORDS),
            token_pattern=r'\b[a-záéíóúñ][a-záéíóúñ_]+\b',
            min_df=3,
            max_df=0.9
        ))
    ])
    
    # Procesamiento numérico
    numeric_features = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # ColumnTransformer
    preprocessor = ColumnTransformer([
        ('text', text_features, 'Headline'),
        ('num', numeric_features, ['longitud', 'mayusculas', 'exclamaciones', 
                                 'interrogaciones', 'prop_mayusculas', 
                                 'densidad_puntuacion'] + 
                                 [f'sens_{w}' for w in ['descubren', 'impactante', 
                                                       'escándalo', 'polémica', 'revelan']])
    ])
    
    # Pipeline completo con XGBoost
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42
        ))
    ])
    
    return pipeline

def entrenar_modelo_optimizado(X_train, y_train):
    """Búsqueda de hiperparámetros con GridSearch"""
    pipeline = crear_pipeline()
    
    param_grid = {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [3, 5, 7],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.7, 0.8],
        'preprocessor__text__tfidf__max_features': [10000, 15000],
        'preprocessor__text__tfidf__ngram_range': [(1,2), (1,3)]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=2
    )
    
    print("\nOptimizando hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score en validación: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def analizar_errores(model, X_test, y_test):
    """Análisis detallado de errores"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Crear máscara de errores
    mask_errores = (y_pred != y_test)
    
    # Filtrar solo los errores
    errores = X_test[mask_errores].copy()
    errores['Real'] = y_test[mask_errores]
    errores['Predicho'] = y_pred[mask_errores]
    
    # Obtener probabilidades solo para los errores
    if hasattr(model, 'predict_proba'):
        # Para clasificadores con predict_proba
        proba_errores = y_proba[mask_errores]
        # Tomar la probabilidad máxima de cada predicción
        errores['Confianza'] = np.max(proba_errores, axis=1)
    else:
        # Para clasificadores sin predict_proba (como algunos SVM)
        errores['Confianza'] = np.nan
    
    print("\nAnálisis detallado de errores:")
    print(f"Total errores: {len(errores)}/{len(X_test)} ({len(errores)/len(X_test):.1%})")
    
    if not errores.empty:
        # Mostrar algunos errores
        print("\nAlgunos errores de clasificación:")
        for idx, row in errores.head(5).iterrows():
            print(f"\nReal: {row['Real']}, Predicho: {row['Predicho']}", end=" ")
            if 'Confianza' in row:
                print(f"(Confianza: {row['Confianza']:.2f})")
            print(f"Headline: {row['Headline']}")
    else:
        print("¡No hubo errores de clasificación!")

def main():
    # Configuración inicial
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('modelos', exist_ok=True)

    # 1. Cargar y preparar datos
    datos = cargar_datos()
    datos = agregar_caracteristicas(datos)
    datos_norm = aplicar_normalizacion(datos)
    
    # Guardar corpus normalizado
    ruta_corpus_norm = os.path.join(OUTPUT_FOLDER, 'corpus_normalizado_mejorado.csv')
    datos_norm.to_csv(ruta_corpus_norm, index=False, encoding='utf-8')
    print(f"\nCorpus normalizado guardado en: {ruta_corpus_norm}")

    # 2. Preparar datos para modelado
    X = datos_norm.drop(columns=['Category'])
    y = datos_norm['Category'].map({'Fake': 1, 'True': 0})
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Entrenar modelo optimizado
    model = entrenar_modelo_optimizado(X_train, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test)
    print("\nReporte de clasificación final:")
    print(classification_report(y_test, y_pred))

    # 6. Analizar errores
    analizar_errores(model, X_test, y_test)
    
    # 7. Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo optimizado guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()