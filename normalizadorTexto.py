import pandas as pd
import stanza
from itertools import product

data = pd.read_excel("Rest_Mex_2022.xlsx")
dout = pd.DataFrame(columns=['Content', 'Polarity'])
dout['Polarity'] = data['Polarity']
#dout['Content'] = str(data['Title']) + " " + data['Opinion']
dout['Content'] = data['Title'].astype(str) + " " + data['Opinion'].astype(str)


print(dout['Content'])

def limpiar_texto(texto):
    texto = texto.lower()
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'ä': 'a', 'ë': 'e', 'ï': 'i', 'ö': 'o', 'ü': 'u',
        'ñ': 'n'
    }
    
    for acentuada, normal in reemplazos.items():
        texto = texto.replace(acentuada, normal)
    
    return texto

config = {
    'processors': 'tokenize,mwt,pos,lemma',
    'lang': 'es'
}

nlp = stanza.Pipeline(**config)

def normalizarTexto(texto, limpia, quita_stopwords, lematiza):
    if limpia:
        texto = limpiar_texto(texto)

    try:
        doc = nlp(texto)
        cadenaNorm = []
        for sent in doc.sentences:
            for token in sent.words:
                if quita_stopwords and lematiza:
                    if token.pos not in {'ADP', 'CCONJ', 'DET', 'SCONJ', 'PRON'}:
                        cadenaNorm.append(token.lemma)
                elif quita_stopwords:
                    if token.pos not in {'ADP', 'CCONJ', 'DET', 'SCONJ', 'PRON'}:
                        cadenaNorm.append(token.text)
                elif lematiza:
                    cadenaNorm.append(token.lemma)
                else:
                    cadenaNorm.append(token.text)
        
        return " ".join(cadenaNorm).strip()
    except Exception as e:
        print(f"Error procesando el texto: {e}")
        return ""

# texto = "Día 4) estuve en la alberca y después de un par de bebidas “all inclusive” en un segundo PERDIMOS EL CONOCIMIENTO por completo mi papá y yo, coreeremos"
# print(normalizarTexto(texto, True, True, True))

for i in range(5):
    dout[i, 'Content'] = normalizarTexto(dout['Content'][i], False, True, True)

dout['Content'] = dout['Content'].apply(lambda x: normalizarTexto(str(x), False, True, True))
dout.to_csv(f'./corpusNorm.csv', sep='\t', index=False)
     