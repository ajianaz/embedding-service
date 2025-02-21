import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

def remove_stopwords(text, language='english'):
    """
    Menghapus stopwords dari teks.
    
    Args:
        text (str): Teks input.
        language (str): Bahasa untuk stopwords, default 'english'.
        
    Returns:
        str: Teks tanpa stopwords.
    """
    stop_words = set(stopwords.words(language))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)

def stem_text(text):
    """
    Melakukan stemming pada teks menggunakan PorterStemmer.
    
    Args:
        text (str): Teks input.
        
    Returns:
        str: Teks yang sudah di-stem.
    """
    ps = PorterStemmer()
    tokens = word_tokenize(text)
    stemmed_tokens = [ps.stem(token) for token in tokens]
    return ' '.join(stemmed_tokens)

def lemmatize_text(text):
    """
    Melakukan lemmatization pada teks menggunakan WordNetLemmatizer.
    
    Args:
        text (str): Teks input.
        
    Returns:
        str: Teks yang sudah di-lemmatize.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def optimize_text(text):
    """
    Normalisasi teks: ubah ke lowercase dan hapus tanda/simbol khusus.
    """
    text = text.lower()
    # Hanya menyisakan karakter alfanumerik dan spasi
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text