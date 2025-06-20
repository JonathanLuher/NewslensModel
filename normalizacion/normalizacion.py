import re
import spacy
from unicodedata import normalize
from spacy.lang.es.stop_words import STOP_WORDS

def normalizar(texto, nlp):
    """
    Normalización mejorada con:
    - Manejo de instituciones
    - Conservación de patrones de fake news
    - Marcado especial de verbos y entidades
    """
    # Normalización de caracteres
    texto = normalize('NFKD', texto).encode('ASCII', 'ignore').decode('ASCII')
    
    # Eliminación URLs y menciones
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    
    # Normalización de instituciones
    instituciones = {
        r'\b(rae|real academia española)\b': 'inst_rae',
        r'\b(sep|secretaría de educación pública)\b': 'inst_sep',
        r'\bunam\b': 'inst_unam'
    }
    for pat, repl in instituciones.items():
        texto = re.sub(pat, repl, texto.lower())
    
    # Manejo de signos de puntuación múltiples
    texto = re.sub(r'!{2,}', ' MULTI_EXCL ', texto)
    texto = re.sub(r'\?{2,}', ' MULTI_QUEST ', texto)
    
    # Conservar signos de exclamación/interrogación individuales
    texto = re.sub(r'[^\w\s¡!¿?ñ]', ' ', texto)
    
    # Tokenización avanzada
    doc = nlp(texto)
    tokens = []
    for token in doc:
        if not token.is_space:
            if token.is_punct:
                if token.text in ['!', '¡']:
                    tokens.append('EXCL')
                elif token.text in ['?', '¿']:
                    tokens.append('QUEST')
            else:
                lemma = token.lemma_.lower()
                if len(lemma) > 2 and lemma not in STOP_WORDS:
                    if token.pos_ == 'VERB':
                        tokens.append(f"verb_{lemma}")
                    elif token.ent_type_ in ['PER', 'ORG']:
                        tokens.append(f"ent_{lemma}")
                    else:
                        tokens.append(lemma)
    
    return ' '.join(tokens)