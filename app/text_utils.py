import re

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