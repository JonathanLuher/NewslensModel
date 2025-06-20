import re
import spacy

def normalizar(texto, nlp):
    """
    Normalización 3:
    - Elimina hashtags y signos de puntuación
    - Convierte a minúsculas
    - Tokenización simple (sin lematización)
    - Filtra palabras cortas (<3 caracteres)
    """
    # Eliminar hashtags y puntuación
    texto = re.sub(r'#\w+', '', texto)
    texto = re.sub(r'[^\w\s]', ' ', texto)
    
    # Minúsculas
    texto = texto.lower()
    
    # Tokenización
    doc = nlp(texto)
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    
    # Filtrar palabras cortas
    tokens = [token for token in tokens if len(token) > 2]
    
    return ' '.join(tokens)