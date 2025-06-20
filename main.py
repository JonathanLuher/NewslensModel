import os
import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from normalizacion.normalizacion import normalizar

# Configuración actualizada
CORPUS_PATH = os.path.join('dataset', 'Dataset_fakenews.csv')
OUTPUT_FOLDER = 'dataset/Dataset_normalizado'
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

def crear_representacion_tfidf(X):
    """Crea representación TF-IDF con bigramas"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),  # Unigramas y bigramas
        max_features=5000,    # Límite de features
        stop_words=['que', 'de', 'el', 'la']  # Stopwords adicionales
    )
    X_vec = vectorizer.fit_transform(X)
    return X_vec, vectorizer

def entrenar_modelo_optimizado(X_train, y_train):
    """Optimiza hiperparámetros con GridSearchCV"""
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': [None, 'balanced']
    }
    
    svm = SVC(random_state=0)
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='f1_weighted',
        n_jobs=-1  # Usa todos los cores del CPU
    )
    
    print("\nOptimizando hiperparámetros...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nMejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score en validación: {grid_search.best_score_:.3f}")
    
    return grid_search.best_estimator_

def main():
    # Configurar carpetas
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('modelos', exist_ok=True)

    # 1. Cargar y normalizar datos
    datos = cargar_datos()
    datos_norm = aplicar_normalizacion(datos)
    
    # Guardar corpus normalizado
    ruta_corpus_norm = os.path.join(OUTPUT_FOLDER, 'corpus_normalizado_mejorado.csv')
    datos_norm.to_csv(ruta_corpus_norm, index=False, encoding='utf-8')
    print(f"\nCorpus normalizado guardado en: {ruta_corpus_norm}")

    # 2. Crear representación TF-IDF
    X = datos_norm['Headline']
    y = datos_norm['Category'].map({'Fake': 1, 'True': 0})
    
    X_vec, vectorizer = crear_representacion_tfidf(X)
    joblib.dump(vectorizer, os.path.join('modelos', 'tfidf_vectorizer.pkl'))

    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=0, stratify=y
    )

    # 4. Entrenar modelo optimizado
    model = entrenar_modelo_optimizado(X_train, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test)
    print("\nReporte de clasificación final:")
    print(classification_report(y_test, y_pred))

    # 6. Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo optimizado guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()