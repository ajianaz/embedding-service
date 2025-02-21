import re
import nltk

def download_nltk_resources():
    """
    Memastikan bahwa resource NLTK yang dibutuhkan sudah tersedia.
    Jika belum, resource tersebut akan diunduh.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

# Pastikan resource sudah tersedia sebelum melakukan operasi NLP
download_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Cache global untuk stopwords berdasarkan bahasa agar tidak memuat ulang setiap kali
_stopwords_cache = {}

def get_stopwords(language='english'):
    """
    Mengambil daftar stopwords untuk bahasa tertentu dengan caching.
    
    Args:
        language (str): Bahasa untuk stopwords (default: 'english').
        
    Returns:
        set: Kumpulan stopwords untuk bahasa yang diberikan.
    """
    if language not in _stopwords_cache:
        _stopwords_cache[language] = set(stopwords.words(language))
    return _stopwords_cache[language]

def remove_stopwords(text, language='english'):
    """
    Menghapus stopwords dari teks.
    
    Args:
        text (str): Teks input.
        language (str): Bahasa untuk stopwords (default: 'english').
        
    Returns:
        str: Teks tanpa stopwords.
    """
    sw = get_stopwords(language)
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in sw]
    return ' '.join(filtered_tokens)

def stem_text(text):
    """
    Melakukan stemming pada teks menggunakan PorterStemmer.
    
    Args:
        text (str): Teks input.
        
    Returns:
        str: Teks yang telah di-stem.
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
        str: Teks yang telah di-lemmatize.
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

def optimize_text(text):
    """
    Melakukan normalisasi teks:
    - Mengubah teks menjadi lowercase.
    - Menghapus tanda dan simbol khusus.
    - Menghapus spasi berlebih.
    
    Args:
        text (str): Teks input.
        
    Returns:
        str: Teks yang telah dinormalisasi.
    """
    text = text.lower()
    # Hanya menyisakan karakter alfanumerik dan spasi
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text