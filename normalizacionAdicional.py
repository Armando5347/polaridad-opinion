import re
from spellchecker import SpellChecker


def corregir_texto(texto):
    
    # Inicializa el corrector para el idioma español
    spell = SpellChecker(language='es')
    
    # Dividir el texto en palabras, preservando la puntuación
    palabras = texto.split()
    texto_corregido = []
    
    for palabra in palabras:
        # Extrae signos de puntuación al inicio y al final
        inicio = ''.join(char for char in palabra if not char.isalnum())
        final = ''.join(char for char in reversed(palabra) if not char.isalnum())
        palabra_central = palabra[len(inicio):-len(final) or None]
        
        # Si la palabra está mal escrita, corrige
        if palabra_central and palabra_central not in spell:
            sugerencia = spell.correction(palabra_central)
            palabra_central = sugerencia if sugerencia else palabra_central
        
        # Reconstruye la palabra con signos de puntuación
        texto_corregido.append(f"{inicio}{palabra_central}{final}")
    
    # Une las palabras corregidas en el texto final
    return ' '.join(texto_corregido)



def corregir_repeticiones(texto):
    
    # Elimina repeticiones consecutivas de letras en una palabra (e.g., "caaaarro" -> "carro")
    def corregir_letras_repetidas(palabra):
        return re.sub(r'(.)\1{2,}', r'\1', palabra)

    # Dividir el texto en palabras
    palabras = texto.split()
    palabras_corregidas = []
    ultima_palabra = None

    for palabra in palabras:
        # Corregir letras repetidas en exceso
        palabra_corregida = corregir_letras_repetidas(palabra)

        # Eliminar palabras repetidas consecutivamente
        if palabra_corregida != ultima_palabra:
            palabras_corregidas.append(palabra_corregida)
            ultima_palabra = palabra_corregida

    # Reconstruir el texto corregido
    return ' '.join(palabras_corregidas)


def procesar_texto(texto, aplicar_repeticiones=False, aplicar_ortografia=False):
    
    if aplicar_repeticiones:
        texto = corregir_repeticiones(texto)
    if aplicar_ortografia:
        texto = corregir_texto(texto)
    return texto

# Ejemplo de uso
if __name__ == "__main__":
    texto_original = "Pedimos steak"
    
    # Aplicar ambas correcciones
    texto_corregido = procesar_texto(texto_original, aplicar_repeticiones=True, aplicar_ortografia=True)
    
    print("Texto original:")
    print(texto_original)
    print("\nTexto corregido:")
    print(texto_corregido)
