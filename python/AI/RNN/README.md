# RNN – Recurrent Neural Network

## Mål

Bygge en liten **språkmodell som kan lære mønstre i tekst og generere sin egen tekst**.

* Bli mer kjent med **RNN-er, språkmodeller og praktisk bruk av relevante biblioteker og verktøy** som brukes i AI/ML-utvikling. Jeg trenger derfor ikke implementere alt fra scratch. Poenget er å forstå de viktigste konseptene, samtidig som jeg får erfaring med hvordan slike modeller faktisk bygges og brukes i praksis.
* Bygge prosjektet gradvis, i stedet for å lage hele modellen på én gang. Totalt deler jeg det opp i **7 blokker**, hvor hver blokk introduserer én viktig del av systemet. Hver blokk skal inneholde et lite eksperiment eller en undersøkelse, slik at jeg ikke bare implementerer komponenten, men også forstår hvorfor og hvordan den brukes.

Prosjektet blir omtrent:

```text
Tekst
 ↓
Tekstbehandling
 ↓
Embeddings
 ↓
RNN
 ↓
Språkmodell
 ↓
Trening
 ↓
Tekstgenerering
```

---

## Blokk 1 – Tekstbehandling

**Formål:** Forstå hvordan rå tekst gjøres om til data som kan brukes av et nevralt nettverk, og hvordan ulike tokeniseringsvalg påvirker dette.

Utforske:

* tokenisering og vocabulary
* character-level vs. word-level
* input/target-sekvenser for next-token prediction
* relevante verktøy for tekstbehandling

### Gjennomført

Jeg har laget en liten, gjenbrukbar pipeline for:

```text
Tekst → Tokenisering → Vocabulary → Token-IDer → Input/target-sekvenser
```

Jeg har sammenlignet character-level og word-level tokenisering, blant annet med tanke på vocabulary-størrelse og sekvenslengde. Jeg har også undersøkt hvordan `sequence_length` påvirker treningssekvensene.

For word-level tokenisering har jeg også testet lowercasing og fjerning av punctuation.

### Gjenstår

* undersøke stemming og lemmatization
* undersøke stopword removal
* undersøke `<UNK>` og andre special tokens som `<PAD>`, `<BOS>` og `<EOS>`
* undersøke subword-tokenisering og relevante verktøy

**Sluttresultat:** En gjenbrukbar pipeline som gjør tekst om til ferdige input/target-sekvenser for RNN-modellen.

---

## Blokk 2 – Embeddings

**Mål:** Utforske hvordan tokens kan representeres som vektorer, og hvorfor dette er nyttig.

Undersøke ulike måter å representere tekst på, for eksempel:

* one-hot
* TF-IDF
* embeddings

Visualisere representasjonene, for eksempel med **PCA**, for å undersøke om tokens med lignende betydning eller bruk ender opp nær hverandre.

Deretter lage et enkelt embedding-lag og undersøke hvordan embeddings håndteres i praktiske ML-biblioteker.

**Sluttresultat:** En bedre forståelse av hvordan tekst går fra token IDs til numeriske vektorer som kan brukes av RNN-en.

---

## Blokk 3 – RNN

**Mål:** Forstå hvordan et rekurrent nevralt nettverk kan behandle sekvenser og ta vare på informasjon fra tidligere tidssteg.

Bygge videre på forståelsen fra `NN`-prosjektet og undersøke:

* hidden state
* recurrent weights
* tidssteg
* hvordan informasjon føres videre gjennom en sekvens
* Backpropagation Through Time (BPTT)

Starte med en **svært enkel numerisk RNN**, uten tekst eller embeddings, slik at selve mekanismen kan forstås isolert.

```text
x₁ → h₁
      ↓
x₂ → h₂
      ↓
x₃ → h₃
```

Jeg kan også visualisere hidden states for å undersøke hvordan representasjonen endrer seg gjennom sekvensen.

**Sluttresultat:** En fungerende og forstått RNN som kan behandle en sekvens.

---
## Blokk 4A – Intent classification

**Mål:** Bruke RNN-en til en annen NLP-oppgave enn språkmodellering:
klassifisering av tekst i forhåndsdefinerte intents.

Bygge:

Tekst
 ↓
Tokenisering
 ↓
Embedding
 ↓
RNN
 ↓
Classification layer
 ↓
Intent

## Blokk 4B – Språkmodell

**Mål:** Koble RNN-en til tekst og gjøre den om til en faktisk språkmodell.

Bygge:

```text
Token IDs
 ↓
Embedding
 ↓
RNN
 ↓
Output-lag
 ↓
Softmax
 ↓
Neste-token-prediksjon
```

Introdusere:

* output-lag
* softmax
* cross-entropy loss
* next-token prediction

Undersøke hvordan modellens sannsynligheter endrer seg når den får mer kontekst.

**Sluttresultat:** En språkmodell som kan ta inn en sekvens og gi sannsynligheter for neste token.

---

## Blokk 5 – Trening

**Mål:** Trene hele modellen på faktisk tekst og undersøke hvordan RNN-en lærer.

Sette sammen:

```text
tekst
 ↓
input/target-sekvenser
 ↓
forward pass
 ↓
prediction
 ↓
loss
 ↓
BPTT
 ↓
gradient descent
 ↓
oppdaterte parametere
```

Eksperimentere med blant annet:

* learning rate
* sequence length
* antall hidden units
* trenings- og valideringsdata
* loss over tid

Undersøke problemer som **vanishing/exploding gradients** og hvordan de påvirker treningen.

Bli kjent med relevante biblioteker for trening, datasett, validering og lagring av modeller.

**Sluttresultat:** En faktisk trent RNN-basert språkmodell.

---

## Blokk 6 – Tekstgenerering

**Mål:** Bruke den trente modellen til å generere tekst.

Gi modellen en startsekvens:

```text
"The cat"
```

og la den predikere ett token om gangen:

```text
"The cat"
"The cat sat"
"The cat sat on"
"The cat sat on the"
...
```

Eksperimentere med:

* random sampling
* temperature
* ulike prompts
* genereringslengde

Undersøke hvordan ulike samplingstrategier påvirker resultatet.

**Sluttresultat:** En liten språkmodell som faktisk kan generere tekst.

---

## Blokk 7 – Videreutvikling

**Mål:** Utforske RNN-ens begrensninger og forstå hvorfor nyere arkitekturer ble utviklet.

Undersøke problemer med vanlige RNN-er, spesielt:

* lange sekvenser
* vanishing/exploding gradients
* begrenset memory

Deretter utforske utviklingen:

```text
RNN
 ↓
LSTM
 ↓
GRU
 ↓
Transformer
```

Jeg
