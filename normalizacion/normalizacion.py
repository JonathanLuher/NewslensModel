import re
import spacy
from unicodedata import normalize

def normalizar(texto, nlp):
    """
    Normalización mejorada:
    - Elimina URLs, menciones, hashtags
    - Conserva signos de exclamación/interrogación
    - Normaliza caracteres acentuados
    - Maneja nombres propios combinados
    - Lematización y filtrado de stopwords
    """
    # Normalizar caracteres unicode (ej. á -> a)
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    
    # Eliminar menciones y hashtags
    texto = re.sub(r'@\w+|#\w+', '', texto)
    
    # Conservar signos de exclamación/interrogación
    texto = re.sub(r'[^\w\s¡!¿?ñ]', ' ', texto.lower())
    
    # Manejar nombres propios combinados (YORDI ROSADO -> yordi_rosado)
    texto = re.sub(r'\b([A-ZÁÉÍÓÚÑ]{2,}\s[A-ZÁÉÍÓÚÑ]{2,})\b', 
                  lambda m: m.group(1).lower().replace(' ', '_'), texto)
    
    # Tokenización y lematización
    doc = nlp(texto)
    tokens = []
    
    for token in doc:
        if not token.is_punct and not token.is_space:
            # Manejar signos de puntuación especiales
            if token.text in ['!', '¡']:
                tokens.append('EXCL_')
            elif token.text in ['?', '¿']:
                tokens.append('QUEST_')
            # Conservar negaciones
            elif token.lower_ in ['no', 'ni', 'nunca']:
                tokens.append('NOT_')
            else:
                lemma = token.lemma_.lower()
                if len(lemma) > 2:
                    tokens.append(lemma)
    
    return ' '.join(tokens)