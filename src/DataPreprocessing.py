"""
DataPreprocessing.py
Módulo de preprocesamiento de texto para clasificador de ODS
"""

import nltk
import re

# Descargar recursos necesarios de NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.stem import WordNetLemmatizer
from nltk import RegexpTokenizer
from nltk.corpus import stopwords


# Inicializar herramientas fuera de la función para que sean serializables con joblib
lemmatizer = WordNetLemmatizer()
stopwords_es = set(stopwords.words('spanish'))


def text_preprocess(text: str) -> str:
    """
    Aplica el pipeline de preprocesamiento de texto:
    1. Tokenización (RegexpTokenizer - elimina puntuación y caracteres especiales)
    2. Eliminación de stopwords en español
    3. Lematización (WordNetLemmatizer)

    Args:
        text (str): Texto crudo de entrada

    Returns:
        str: Texto preprocesado como string de tokens unidos por espacios
    """
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word not in stopwords_es]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
