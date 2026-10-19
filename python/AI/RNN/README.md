# RNN – Recurrent Neural Networks

## Mål
Utforske de viktigste byggesteinene bak RNN-er gjennom praktiske eksperimenter, og til slutt bygge en liten modell som lærer å forutsi neste token i en tekstsekvens.

## Block 1 – Tokenization
I blokk 1 utforsker jeg det første steget innen dataprosessering: å dele opp tekst i tokens og lage et *vocabulary* som mapper hvert token til en ID. 
I tillegg implementerer jeg subword-tokenisering med Byte Pair Encoding (BPE)

**Eksempel:**
```python
"the cat sat"
→ ["the", "cat", "sat"] # tokens
→ [0, 1, 2]             # ID's

# the mapping: token:ID
vocabulary = {
    "the": 0,
    "cat": 1,
    "sat": 2
}

# BPE merge frequent character pairs into subwords
"the cat sat"
→ ["the", "cat", "s", "at"]
```

## Block 2 – Embeddings
I blokk 2 utforsker jeg hvordan token-ID-er representeres som numeriske vektorer som et nevralt nettverk kan behandle. Jeg undersøker embedding-matriser, direkte oppslag og hvordan embeddings brukes som input til en modell.

**Eksempel:**
```python
token_id = 1  # "cat"

embedding = embedding_matrix[token_id]
position = positional_encoding(position_id, embedding_dim)

# embedding    [0.2, -0.4, 0.7]
# posisjon     [-0.3, 0.8, 0.1]
# kombinert    [-0.1, 0.4, 0.8]
```

**NB! Positional Vectors** representerer hvor i en sekvens et token befinner seg.  
I Transformers kombineres posisjon + embedding, slik at modellen kan ta hensyn til rekkefølgen på tokens. RNN bruker vanligvis ikke det, men er inkludert likevel.

## Block 3 – RNN Fundamentals
I blokk 3 bygger jeg en enkel RNN som bruker embeddings (fra blokk 2) til å behandle tekstsekvenser og forutsi neste token.  
Jeg utforser _hidden states_, vekter, bias, aktiveringsfunksjoner, loss, og _Backpropagation Through Time (BPTT)_, samt viser én enkel treningsppdatering.

**Eksempel:**
```python
# embedding
x = embedding_matrix[token_id]

# Activation for hidden state
h = np.tanh(
    W_x @ x + W_h @ h_previous + b
)

# predict next token
probabilities = softmax(W_y @ h + b_y)

# trening:
Input:       "the cat sat on the"
Target:      "mat"

Prediksjon:  "sat"
Loss:        måler hvor feil prediksjonen er

→ Oppdater vektene
→ Gjenta treningen
```

## Block 4 – Training
I blokk 4 trener jeg RNN-modellen på et lite tekstkorpus og kombinerer tokenisering, embeddings, og trening i én sammenhengende pipeline.  
Jeg undersøøker hvordan modellen lærer fra _flere_ eksempler og evaluerer prediksjonene.

**Eksempel:**
```text
Input:       "the cat sat"          target: "on"
Prediksjon:  "on"

Input:       "the cat sat on"       target: "the"
Prediksjon:  "the"

Input:       "the cat sat on the"   target: "mat"
Prediksjon:  "mat"
```

*Eksemplene viser ønsket oppførsel, ikke garanterte prediksjoner fra modellen.*