# RNN – Recurrent Neural Networks

## Mål

Utforske de viktigste byggesteinene bak RNN-er gjennom praktiske eksperimenter, og til slutt bygge en liten modell som lærer å forutsi neste token i en tekstsekvens.

## Block 1 – Tokenization

I blokk 1 utforsker jeg det første steget innen dataprosessering: å dele opp tekst i tokens og lage et *vocabulary* som mapper hvert token til en ID.

I tillegg implementerer jeg subword-tokenisering med Byte Pair Encoding (BPE), som lærer å slå sammen hyppige tegnpar til større subwords.

**Eksempel:**

```python
"the cat sat"
→ ["the", "cat", "sat"]  # tokens
→ [0, 1, 2]              # IDs

# Mapping: token → ID
vocabulary = {
    "the": 0,
    "cat": 1,
    "sat": 2
}

# BPE lærer å slå sammen pair som forekommer ofte
"The cat was happy. The cat was unhappy."
→ ["▁the", "▁cat", "▁was", "▁happy", ".", 
   "▁the", "▁cat", "▁was", "▁un", "happy", "."]

# Vocabulary inneholder unike tokens
→ ["▁the", "▁cat", "▁was", "▁happy", ".", "▁un", ...]
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

**NB! Positional Vectors** representerer hvor i en sekvens et token befinner seg. I Transformers kombineres posisjon og embedding, slik at modellen kan ta hensyn til rekkefølgen på tokens. RNN-er bruker vanligvis ikke positional encodings, fordi rekkefølgen allerede behandles gjennom de rekurrente hidden states. Konseptet er inkludert for å illustrere forskjellen mellom modellarkitekturer.

## Block 3 – RNN Fundamentals

I blokk 3 bygger jeg en enkel RNN som bruker embeddings fra blokk 2 til å behandle tekstsekvenser og forutsi neste token.

Jeg utforsker *hidden states*, vekter, bias, aktiveringsfunksjoner, loss og *Backpropagation Through Time (BPTT)*, samt viser én enkel treningsoppdatering.

**Eksempel:**

```python
# Embedding
x = embedding_matrix[token_id]

# Beregn hidden state
h = np.tanh(
    W_x @ x + W_h @ h_previous + b
)

# Prediker neste token
probabilities = softmax(W_y @ h + b_y)

# Trening
Input:       "the cat sat on the"
Target:      "mat"

Prediksjon:  "sat"
Loss:        Måler hvor feil prediksjonen er

→ Oppdater vektene
→ Gjenta treningen
```

## Block 4 – Training

I blokk 4 trener jeg RNN-modellen på et lite tekstkorpus og kombinerer tokenisering, embeddings og trening i én sammenhengende pipeline.

Jeg undersøker hvordan modellen lærer fra flere eksempler, evaluerer prediksjonene på trenings- og testdata og eksperimenterer med tekstgenerering.

**Eksempel:**

```text
Input:       "the cat sat"          Target: "on"
Prediksjon:  "on"

Input:       "the cat sat on"       Target: "the"
Prediksjon:  "the"

Input:       "the cat sat on the"   Target: "mat"
Prediksjon:  "mat"
```

*Eksemplene viser ønsket oppførsel: modellen lærer å forutsi neste token i en sekvens basert på tidligere tokens.*

## Block 5 - Practical implementation

I blokk 5 implementerer jeg en enkel RNN-modell i Python ved hjelp av både numpy og PyTorch.  
Jeg viser hvordan man kan bygge en RNN fra bunnen av, samt hvordan man kan bruke PyTorchs innebygde RNN-moduler for å forenkle implementasjonen.

**Eksempel**:

```python
# RNN with pyTorch
```