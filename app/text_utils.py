import re

import re

def optimize_text(text, replace_symbols=False):
    """
    Melakukan normalisasi teks:
    - Mengubah teks menjadi lowercase.
    - (Opsional) Jika replace_symbols bernilai True (atau "true"), mengganti simbol-simbol tertentu dengan kata yang sesuai.
    - Menghapus tanda dan simbol khusus yang tersisa.
    - Menghapus spasi berlebih.
    
    Args:
        text (str): Teks input.
        replace_symbols (bool or str, optional): Jika True atau "true" (case-insensitive), lakukan penggantian simbol dengan kata. 
            Default adalah False.
        
    Returns:
        str: Teks yang telah dinormalisasi dan dioptimasi.
    """
    text = text.lower()
    
    # Konversi parameter replace_symbols jika berupa string
    if isinstance(replace_symbols, str):
        replace_symbols = replace_symbols.lower() == "true"
    
    if replace_symbols:
        # Dictionary penggantian simbol dengan padanan kata
        replacements = {
            '%': ' persen ',
            '$': ' dolar ',
            '€': ' euro ',
            '£': ' pound ',
            '₹': ' rupiah ',
            '₩': ' won ',
            '₽': ' ruble ',
            '&': ' dan ',
            '@': ' at ',
            '#': ' hash ',
            '*': ' bintang ',
            '+': ' tambah ',
            '=': ' sama dengan ',
            '>': ' lebih besar dari ',
            '<': ' lebih kecil dari ',
            # '~': ' tilde',
            # '^': ' topi',
            # '_': ' garis bawah',
            # '|': ' garis vertical',
            # '\\': ' backslash',
            # '/': ' slash',
            # '?': ' tanya',
            # '!': ' seru',
            # ':': ' titik dua',
            # ';': ' titik koma',
            # ',': ' koma',
            # '.': ' titik',
            # '\'': ' kutip tunggal',
            # '"': ' kutip ganda',
            # '(': ' buka kurung',
            # ')': ' tutup kurung',
            # '[': ' buka kurung siku',
            # ']': ' tutup kurung siku',
            # '{': ' buka kurung kurawal',
            # '}': ' tutup kurung kurawal',
            # '`': ' backtick',
            # '…': ' titik-titik',
            # '–': ' strip',
            # '—': ' strip panjang',
            # 'º': ' derajat',
            # 'ª': ' ordinal',
        }
        for symbol, word in replacements.items():
            text = text.replace(symbol, word)
    
    # Hapus karakter selain huruf, angka, dan spasi
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Hilangkan spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text