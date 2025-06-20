import os
import pandas as pd
import spacy
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from spacy.lang.es.stop_words import STOP_WORDS as SPANISH_STOP_WORDS
from normalizacion.normalizacion import normalizar

# Configuración
CORPUS_PATH = os.path.join('dataset', 'Dataset_fakenews.csv')
OUTPUT_FOLDER = 'corpus/normalizacion_mejorada'
MODEL_PATH = os.path.join('modelos', 'svm_optimizado.pkl')

def cargar_datos():
    """Carga el corpus original"""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"No se encontró el archivo en {CORPUS_PATH}")
    return pd.read_csv(CORPUS_PATH, encoding='utf-8')

def aplicar_normalizacion(datos):
    """Aplica la normalización mejorada a los headlines"""
    nlp = spacy.load('es_core_news_sm')
    datos_normalizados = datos.copy()
    datos_normalizados['Headline'] = datos_normalizados['Headline'].apply(
        lambda x: normalizar(str(x), nlp))
    return datos_normalizados

def agregar_caracteristicas(df):
    """Agrega características adicionales basadas en el texto"""
    # Longitud del título
    df['longitud'] = df['Headline'].apply(len)
    
    # Conteo de mayúsculas
    df['mayusculas'] = df['Headline'].apply(lambda x: sum(1 for c in x if c.isupper()))
    
    # Presencia de signos de exclamación/interrogación
    df['exclamaciones'] = df['Headline'].str.count(r'[¡!]')
    df['interrogaciones'] = df['Headline'].str.count(r'[¿?]')
    
    return df

def crear_pipeline():
    """Crea el pipeline completo de procesamiento y modelado"""
    # Pipeline para características de texto
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words=list(SPANISH_STOP_WORDS),
            token_pattern=r'\b[a-záéíóúñ][a-záéíóúñ_]+\b'
        ))
    ])
    
    # Pipeline para características numéricas
    numeric_features = Pipeline([
        ('scaler', StandardScaler())
    ])
    
    # Combinar transformers
    preprocessor = ColumnTransformer([
        ('text', text_features, 'Headline'),
        ('num', numeric_features, ['longitud', 'mayusculas', 'exclamaciones', 'interrogaciones'])
    ])
    
    # Pipeline completo
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', SVC(
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            random_state=42
        ))
    ])
    
    return pipeline

def entrenar_modelo_optimizado(X_train, y_train):
    """Optimiza hiperparámetros con GridSearchCV"""
    pipeline = crear_pipeline()
    
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf'],
        'preprocessor__text__tfidf__max_features': [5000, 10000],
        'preprocessor__text__tfidf__ngram_range': [(1,2), (1,3)]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    print("\nOptimizando hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score en validación: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def analizar_errores(model, X_test, y_test):
    """Analiza muestras mal clasificadas"""
    y_pred = model.predict(X_test)
    errores = X_test[y_pred != y_test].copy()  # Hacer una copia explícita
    y_test_errores = y_test[y_pred != y_test]
    y_pred_errores = y_pred[y_pred != y_test]
    
    print(f"\nMuestras mal clasificadas ({len(errores)}/{len(X_test)}):")
    
    # Usamos iterrows() para obtener tanto el índice como la fila
    for (idx, row), real, pred in zip(errores.iterrows(), y_test_errores, y_pred_errores):
        print(f"\nReal: {real}, Predicho: {pred}")
        print(f"Headline: {row['Headline']}")
        if 'Text' in row:
            print(f"Primeras 100 chars: {row['Text'][:100]}...")

def main():
    # Configurar carpetas
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('modelos', exist_ok=True)

    # 1. Cargar datos
    datos = cargar_datos()
    
    # 2. Agregar características adicionales
    datos = agregar_caracteristicas(datos)
    
    # 3. Normalización mejorada
    datos_norm = aplicar_normalizacion(datos)
    
    # Guardar corpus normalizado
    ruta_corpus_norm = os.path.join(OUTPUT_FOLDER, 'corpus_normalizado_mejorado.csv')
    datos_norm.to_csv(ruta_corpus_norm, index=False, encoding='utf-8')
    print(f"\nCorpus normalizado guardado en: {ruta_corpus_norm}")

    # 4. Preparar datos para modelado
    X = datos_norm[['Headline', 'longitud', 'mayusculas', 'exclamaciones', 'interrogaciones']]
    y = datos_norm['Category'].map({'Fake': 1, 'True': 0})
    
    # 5. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Entrenar modelo optimizado
    model = entrenar_modelo_optimizado(X_train, y_train)
    
    # 7. Evaluar
    y_pred = model.predict(X_test)
    print("\nReporte de clasificación final:")
    print(classification_report(y_test, y_pred))

    # 8. Analizar errores
    analizar_errores(model, X_test, y_test)
    
    # 9. Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo optimizado guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()