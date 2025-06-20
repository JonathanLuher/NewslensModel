import os
import pandas as pd
import spacy
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from normalizacion.normalizacion import normalizar

# Configuración (actualizada con el nuevo dataset)
CORPUS_PATH = os.path.join('dataset', 'Dataset_fakenews.csv')
OUTPUT_FOLDER = 'corpus/normalizacion3'
MODEL_PATH = os.path.join('modelos', 'svm_unigram_binary_headline.pkl')

def cargar_datos():
    """Carga el corpus original"""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"No se encontró el archivo del corpus en {CORPUS_PATH}")
    return pd.read_csv(CORPUS_PATH)

def aplicar_normalizacion(datos):
    """Aplica la normalización 3 al texto (ahora en la columna Headline)"""
    nlp = spacy.load('es_core_news_sm')
    datos_normalizados = datos.copy()
    datos_normalizados['Headline'] = datos_normalizados['Headline'].apply(
        lambda x: normalizar(str(x), nlp))
    return datos_normalizados

def crear_representacion_unigram_binary(X):
    """Crea representación unigram-binary"""
    vectorizer = CountVectorizer(binary=True, ngram_range=(1,1))
    X_vec = vectorizer.fit_transform(X)
    return X_vec, vectorizer

def entrenar_modelo(X_train, y_train):
    """Entrena el modelo SVM con los mejores parámetros encontrados"""
    svm = SVC(
        C=10, 
        kernel='linear', 
        random_state=0
    )
    svm.fit(X_train, y_train)
    return svm

def main():
    # Configurar carpetas
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('modelos', exist_ok=True)

    # 1. Cargar y normalizar datos
    datos = cargar_datos()
    datos_norm = aplicar_normalizacion(datos)
    
    # Guardar corpus normalizado
    ruta_corpus_norm = os.path.join(OUTPUT_FOLDER, 'corpus_normalizado3_headline.csv')
    datos_norm.to_csv(ruta_corpus_norm, index=False)
    print(f"Corpus normalizado guardado en: {ruta_corpus_norm}")

    # 2. Crear representación unigram-binary usando Headline
    X = datos_norm['Headline']  # Ahora usamos Headline en lugar de Teaser Text
    y = datos_norm['Category']  # Usamos Category (Fake/True) como etiqueta
    
    # Convertir etiquetas a binario (Fake = 1, True = 0)
    y = y.map({'Fake': 1, 'True': 0})
    
    X_vec, vectorizer = crear_representacion_unigram_binary(X)
    
    # Guardar vectorizer para uso futuro
    joblib.dump(vectorizer, os.path.join('modelos', 'unigram_binary_vectorizer_headline.pkl'))

    # 3. Dividir datos (80% train - 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=0, stratify=y
    )

    # 4. Entrenar modelo SVM
    print("\nEntrenando modelo SVM...")
    model = entrenar_modelo(X_train, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test)
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred))

    # 6. Guardar modelo
    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo SVM guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()