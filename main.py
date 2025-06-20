import os
import pandas as pd
import spacy
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from spacy.lang.es.stop_words import STOP_WORDS
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2

# Importar la función normalizar correctamente
from normalizacion.normalizacion import normalizar

# Configuración
CORPUS_PATH = os.path.join('dataset', 'Dataset_fakenews.csv')
OUTPUT_FOLDER = 'dataset/Dataset_normalizado'
MODEL_PATH = os.path.join('modelos', 'optimized_model.pkl')

# Definir función de tokenización fuera de crear_pipeline para poder serializarla
def tokenize_text(text):
    return text.split()

def cargar_datos():
    """Carga el corpus original con verificación de calidad"""
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(f"No se encontró el archivo en {CORPUS_PATH}")
    
    datos = pd.read_csv(CORPUS_PATH, encoding='utf-8')
    
    # Verificación básica de calidad de datos
    if datos.isnull().sum().sum() > 0:
        print("Advertencia: Hay valores nulos en el dataset")
        datos = datos.dropna(subset=['Text', 'Category'])  # Cambio a columna Text
    
    # Filtrar textos muy cortos
    datos = datos[datos['Text'].str.len() > 50]
    
    return datos

def aplicar_normalizacion(datos):
    """Aplica la normalización mejorada al texto completo"""
    nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
    datos_normalizados = datos.copy()
    
    # Preprocesamiento adicional antes de normalizar
    datos_normalizados['Text'] = datos_normalizados['Text'].astype(str)
    
    # Aplicar normalización al texto completo
    print("Normalizando textos (esto puede tomar tiempo)...")
    datos_normalizados['Text'] = [normalizar(texto, nlp) for texto in datos_normalizados['Text']]
    
    return datos_normalizados

def agregar_caracteristicas(df):
    """Feature engineering avanzado para texto completo"""
    # Características básicas
    df['longitud'] = df['Text'].apply(len)
    df['num_palabras'] = df['Text'].apply(lambda x: len(x.split()))
    df['palabras_unicas'] = df['Text'].apply(lambda x: len(set(x.split())))
    df['palabras_unicas_ratio'] = df['palabras_unicas'] / df['num_palabras']
    df['mayusculas'] = df['Text'].apply(lambda x: sum(1 for c in x if c.isupper()))
    
    # Características de puntuación
    df['exclamaciones'] = df['Text'].str.count(r'[¡!]')
    df['interrogaciones'] = df['Text'].str.count(r'[¿?]')
    df['puntuacion_total'] = df['exclamaciones'] + df['interrogaciones']
    
    # Características de contenido emocional/sensacionalista
    sensacionalistas = [
        'urgente', 'exclusivo', 'impactante', 'revelación', 
        'escándalo', 'impacto', 'descubre', 'revela', 'sorprende',
        'asombroso', 'increíble', 'aterrador', 'alerta', 'peligro',
        'shock', 'bomba', 'explosivo', 'oculto', 'censurado', 'conspiración'
    ]
    df['sensacionalistas'] = df['Text'].str.count(
        r'\b(' + '|'.join(sensacionalistas) + r')\b')
    
    # Características de URL (si está presente en el texto)
    df['contiene_url'] = df['Text'].str.contains(r'http[s]?://', na=False).astype(int)
    
    # Características de entidades nombradas (aproximación simple)
    df['num_entidades'] = df['Text'].str.count(r'ent_\w+')
    
    # Característica de polaridad (positiva/negativa)
    palabras_positivas = ['bueno', 'excelente', 'maravilloso', 'genial', 'positivo']
    palabras_negativas = ['malo', 'terrible', 'horrible', 'pésimo', 'negativo']
    df['polaridad_pos'] = df['Text'].str.count(r'\b(' + '|'.join(palabras_positivas) + r')\b')
    df['polaridad_neg'] = df['Text'].str.count(r'\b(' + '|'.join(palabras_negativas) + r')\b')
    
    # Característica de citas (indicador de veracidad)
    df['num_citas'] = df['Text'].str.count(r'"(.*?)"')
    
    # Característica de nombres propios (mayúsculas sostenidas)
    df['nombres_propios'] = df['Text'].str.count(r'\b[A-Z][a-z]+\b')
    
    return df

def crear_pipeline():
    """Pipeline optimizado para análisis de texto completo"""
    # Procesamiento de texto mejorado
    text_features = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_text,
            min_df=5,
            max_df=0.85,
            ngram_range=(1, 2),  # Reducido a bigramas por el texto más largo
            max_features=15000,   # Aumentado para capturar más información
            sublinear_tf=True     # Suavizar la frecuencia de términos
        )),
        ('feature_selection', SelectKBest(chi2, k=8000))
    ])
    
    # Procesamiento numérico
    numeric_features = Pipeline([
        ('scaler', MinMaxScaler())
    ])
    
    # ColumnTransformer - Cambiado a usar 'Text' en lugar de 'Headline'
    preprocessor = ColumnTransformer([
        ('text', text_features, 'Text'),  # Cambio clave aquí
        ('num', numeric_features, ['longitud', 'num_palabras', 'palabras_unicas', 
                                 'palabras_unicas_ratio', 'mayusculas', 'exclamaciones', 
                                 'interrogaciones', 'puntuacion_total', 'sensacionalistas',
                                 'contiene_url', 'num_entidades', 'polaridad_pos', 
                                 'polaridad_neg', 'num_citas', 'nombres_propios'])
    ])
    
    # Modelo base XGBoost optimizado
    xgb_model = XGBClassifier(
        n_estimators=400,  # Aumentado para texto más complejo
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.7,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=0.2,
        eval_metric='logloss',
        random_state=42,
        use_label_encoder=False
    )
    
    # Modelo SVM para stacking
    svm_model = SVC(
        C=1.5,  # Aumentado para texto más complejo
        kernel='linear',
        probability=True,
        random_state=42
    )
    
    # Modelo Random Forest para diversidad
    rf_model = RandomForestClassifier(
        n_estimators=300,  # Aumentado para texto más complejo
        max_depth=12,
        min_samples_split=3,
        random_state=42
    )
    
    # Ensemble con Stacking
    ensemble = StackingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('svm', svm_model),
            ('rf', rf_model)
        ],
        final_estimator=LogisticRegression(
            penalty='l2',
            C=0.2,
            solver='liblinear',
            max_iter=1500
        ),
        stack_method='predict_proba',
        passthrough=True
    )
    
    # Usamos ImbPipeline para manejar SMOTE correctamente
    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
        ('clf', ensemble)
    ])

def evaluar_modelo_cruzado(model, X, y):
    """Evaluación con validación cruzada estratificada"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=1)
    except Exception as e:
        print(f"Error en validación cruzada: {str(e)}")
        return 0
    
    print("\nResultados de Validación Cruzada (5 folds):")
    print(f"F1-score promedio: {scores.mean():.2f} (+/- {scores.std():.2f})")
    print(f"Scores individuales: {scores}")
    
    return scores.mean()

def main():
    # Configuración inicial
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs('modelos', exist_ok=True)

    # 1. Cargar y preparar datos
    print("Cargando y preparando datos...")
    datos = cargar_datos()
    datos = agregar_caracteristicas(datos)
    datos_norm = aplicar_normalizacion(datos)
    
    # Guardar corpus normalizado
    ruta_corpus_norm = os.path.join(OUTPUT_FOLDER, 'Dataset_normalizado.csv')
    datos_norm.to_csv(ruta_corpus_norm, index=False, encoding='utf-8')
    print(f"\nCorpus normalizado guardado en: {ruta_corpus_norm}")

    # 2. Preparar datos para modelado
    X = datos_norm.drop(columns=['Category'])
    y = datos_norm['Category'].map({'Fake': 1, 'True': 0})
    
    # Verificar balance de clases
    print("\nDistribución de clases:")
    print(y.value_counts(normalize=True))
    
    # 3. Dividir datos para evaluación
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Crear y evaluar modelo
    print("\nCreando modelo para análisis de texto completo...")
    pipeline = crear_pipeline()
    
    # 5. Evaluación con validación cruzada
    print("\nEvaluando modelo con validación cruzada...")
    cv_score = evaluar_modelo_cruzado(pipeline, X_train, y_train)
    
    # 6. Entrenar modelo final
    print("\nEntrenando modelo final...")
    pipeline.fit(X_train, y_train)
    
    # 7. Evaluación final
    print("\nEvaluando modelo en conjunto de prueba...")
    y_pred = pipeline.predict(X_test)
    print("\nReporte de clasificación final:")
    print(classification_report(y_test, y_pred))
    print(f"\nAccuracy en test: {accuracy_score(y_test, y_pred):.2f}")
    print(f"F1-score en test: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    
    # 8. Guardar modelo
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nModelo optimizado guardado en: {MODEL_PATH}")

if __name__ == "__main__":
    main()