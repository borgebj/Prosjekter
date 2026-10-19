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

# mapping: token:ID
vocabulary = {
    "the": 0,
    "cat": 1,
    "sat": 2
}

# BPE merger ord som forekommer ofte i subwords
"The cat was happy. The cat was unhappy."
→ ["the", "cat", "was", "happy", ".", "the", "cat", "was", "un", "happy", "."]

# Unike tokens (nytt vocabulary)
→ ["the", "cat", "was", "happy", ".", "un"]
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
I blokk 3 utforsker jeg hvordan en RNN behandler én input om gangen og viderefører informasjon fra tidligere tidssteg gjennom en *hidden state*. Jeg bygger videre på grunnleggende prinsipper fra nevrale nettverk, som vekter, bias og aktiveringsfunksjoner.

**Eksempel:**
```python
# embedding
x = embedding_matrix[token_id]

# Activation for hidden state
h = np.tanh(
    W_x @ x + W_h @ h_previous + b
)
```

## Block 4 – Training
I blokk 4 utforsker jeg hvordan en RNN lærer ved å sammenligne prediksjoner med riktige svar og justere vektene for å redusere feilen. Jeg undersøker loss, gradient descent og hovedideen bak *Backpropagation Through Time (BPTT)*.

**Eksempel:**
```text
Input:       "the cat sat"
Target:      "on"

Prediksjon:  "mat"
Loss:        måler hvor feil prediksjonen er

→ Oppdater vektene
→ Gjenta treningen
```

## Block 5 – RNN Prototype
I blokk 5 kombinerer jeg tokenisering, embeddings, RNN og trening til én fungerende prototype. Modellen trenes på et lite tekstkorpus og forsøker å forutsi neste token basert på tidligere tokens.

**Eksempel:**
```text
Input:       "the cat sat"
Prediksjon:  "on"

Input:       "the cat sat on"
Prediksjon:  "the"

Input:       "the cat sat on the"
Prediksjon:  "mat"
```

*Eksemplene viser ønsket oppførsel, ikke garanterte prediksjoner fra modellen.*
