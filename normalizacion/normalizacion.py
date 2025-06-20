import re
import spacy
from unicodedata import normalize
from spacy.lang.es.stop_words import STOP_WORDS

def normalizar(texto, nlp):
    # Normalización avanzada de instituciones
    instituciones = {
        r'\b(rae|real academia española)\b': 'inst_rae',
        r'\b(sep|secretaría de educación pública)\b': 'inst_sep',
        r'\bunam\b': 'inst_unam',
        r'\b(ine|instituto nacional electoral)\b': 'inst_ine'
    }
    
    # Convertir a minúsculas después de marcar instituciones
    texto = texto.lower()
    for pat, repl in instituciones.items():
        texto = re.sub(pat, repl, texto)
    
    # Manejo de números (convertir a marcadores)
    texto = re.sub(r'\d+', 'NUM', texto)
    
    # Detección de frases sensacionalistas
    sensacionalistas = {
        r'\b(descubr[íi]|revel[óo]|impactante|esc[áa]ndalo)\b': 'sensacional',
        r'!{2,}': 'MULTI_EXCL',
        r'\?{2,}': 'MULTI_QUEST'
    }
    for pat, repl in sensacionalistas.items():
        texto = re.sub(pat, repl, texto)
    
    # Tokenización avanzada con spaCy
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
                lemma = token.lemma_
                if token.pos_ in ['VERB', 'AUX']:
                    tokens.append(f"verb_{lemma}")
                elif token.ent_type_:
                    tokens.append(f"ent_{token.ent_type_}_{lemma}")
                elif len(lemma) > 2 and lemma not in STOP_WORDS:
                    tokens.append(lemma)
    
    return ' '.join(tokens)