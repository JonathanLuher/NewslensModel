import re
import spacy
from unicodedata import normalize

def normalizar(texto, nlp):
    """
    Normalización mejorada:
    - Elimina URLs, menciones, hashtags
    - Elimina números y puntuación
    - Lematización (en lugar de solo tokenización)
    - Filtra stopwords personalizadas
    - Normaliza caracteres acentuados
    """
    # Normalizar caracteres unicode (ej. á -> a)
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    # Eliminar menciones y hashtags
    texto = re.sub(r'@\w+|#\w+', '', texto)
    
    # Eliminar números
    texto = re.sub(r'\d+', '', texto)
    
    # Eliminar puntuación y caracteres especiales (conserva ñ)
    texto = re.sub(r'[^\w\sñ]|_', ' ', texto)
    
    # Minúsculas
    texto = texto.lower().strip()
    
    # Tokenización y lematización
    doc = nlp(texto)
    
    # Stopwords personalizadas (puedes añadir más)
    custom_stopwords = {"que", "de", "el", "la", "los", "las", "un", "una", "con"}
    
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop 
        and not token.is_punct 
        and not token.is_space
        and token.lemma_ not in custom_stopwords
        and len(token.lemma_) > 2
    ]
    
    return ' '.join(tokens)