# RNN – Recurrent Neural Network

## Mål

Bygge en liten **språkmodell som kan lære mønstre i tekst og generere sin egen tekst**.

TODO  
* Bli mer kjent med **RNN-er, språkmodeller og praktisk bruk av relevante biblioteker og verktøy** som brukes i AI/ML-utvikling. Jeg trenger derfor ikke implementere alt fra scratch. Poenget er å forstå de viktigste konseptene, samtidig som jeg får erfaring med hvordan slike modeller faktisk bygges og brukes i praksis.  
    

* Bygge prosjektet gradvis, i stedet for å lage hele modellen på én gang. Totalt deler jeg det opp i **7 blokker**, hvor hver blokk introduserer én viktig del av systemet. Etter hvert skal blokkene kunne kobles sammen til en fungerende språkmodell.


Prosjektet blir oppdelt i blokker:

```text
Tekst
 ↓
Tokenisering
 ↓
Embeddings
 ↓
RNN
 ↓
Output-lag
 ↓
Softmax
 ↓
Neste-token-prediksjon
 ↓
Tekstgenerering
```

---

## Blokk 1 – Tekstbehandling

**Mål:** Finne ut hvordan jeg gjør vanlig tekst om til data som et nevralt nettverk kan bruke.

Utforske tokenisering og vocabulary, og bli kjent med relevante verktøy for dette.

Eksempel:

```text
"hello"
```

kan bli:

```text
h → 0
e → 1
l → 2
o → 3
```

og dermed:

```text
[0, 1, 2, 2, 3]
```

Må også lage input/target-sekvenser som kan brukes til å lære modellen å forutsi neste tegn/token.

Mulig start:  **character-level**, og så **word-level** senere.

---

## Blokk 2 – Embeddings

**Mål:** Forstå hvordan tokens kan representeres som vektorer i stedet for bare tall.

Et token som:

```text
h → 0
```

skal etter hvert representeres av noe mer som:

```text
h → [0.21, -0.43, 0.72, ...]
```

Finne ut mer av hva embeddings er, hvorfor de brukes, og hvordan et embedding-lag fungerer.

Først lage en enkel embedding selv for å forstå konseptet, og deretter utforske hvordan embeddings håndteres i praktiske ML-biblioteker.

Blir koblingen mellom tekstbehandlingen og selve nevrale nettverket.

---

## Blokk 3 – RNN

**Mål:** Forstå og bruke et rekurrent nevralt nettverk og hvordan det kan behandle sekvenser.

En naturlig utvidelse av den allerede lagde `NN` prosjektet, skal så bygge videre på det.

Forskjellen er at RNN-en tar med seg en **hidden state** fra forrige steg:

```text
x₁ → h₁
      ↓
x₂ → h₂
      ↓
x₃ → h₃
      ↓
x₄ → h₄
```

Finne ut av:
* hidden state
* recurrent weights
* hvordan informasjon føres videre mellom tidssteg
* hvordan sekvenser behandles
* Backpropagation Through Time (BPTT)
* 
---

## Blokk 4 – Språkmodell

**Mål:** Gjøre RNN-en om til en faktisk språkmodell.

I stedet for å bare produsere én verdi, skal modellen forutsi sannsynligheten for hvert mulig neste token.

For eksempel:

```text
Input: "hel"

l → 0.82
o → 0.05
p → 0.02
...
```

Her introduseres:

* output-lag
* softmax
* cross-entropy loss
* next-token prediction

Dette blir omtrent:

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
Sannsynlighet for neste token
```

Så langt kan blokkene brukes hver for seg, men de kan også kobles sammen til en helhetlig modell.

---

## Blokk 5 – Trening

**Mål:** Trene hele modellen på faktisk tekst.

Her skal jeg koble sammen komponentene og lage en ordentlig treningspipeline.

Jeg må blant annet håndtere:

```text
tekst
 ↓
input/target-sekvenser
 ↓
forward pass
 ↓
loss
 ↓
backpropagation / BPTT
 ↓
gradient descent
 ↓
oppdaterte parametere
```

Vil bli kjent med hvordan man faktisk trener modeller i praksis, f.eks gjennom relevante biblioteker for datasett, trening, validering og modell-lagring..

---

## Blokk 6 – Tekstgenerering

**Mål:** Bruke den trente modellen til å faktisk generere tekst.

Skal så kunne gi modellen en starttekst:

```text
"The cat"
```

og la den fortsette:

```text
"The cat s"
"The cat sat"
"The cat sat on"
"The cat sat on the"
...
```

Modellen predikerer ett token om gangen, og det nye tokenet brukes videre som input.

Her kan jeg også eksperimentere med ting som:

* random sampling
* temperature
* hvor lange sekvenser modellen skal generere
* forskjellige prompts

Jeg vil også utforske hvordan tekstgenerering og inference vanligvis håndteres i praktiske ML-verktøy.

Dette er punktet hvor prosjektet faktisk begynner å føles som en liten språkmodell.

---

## Blokk 7 – Videreutvikling

**Mål:** Utforske hvor langt jeg kan ta modellen videre.

Når den vanlige RNN-en fungerer, kan jeg undersøke hvilke problemer den har, spesielt med lange sekvenser og vanishing/exploding gradients.

Derfra kan jeg gå videre til:

```text
RNN
 ↓
LSTM
 ↓
GRU
 ↓
Transformer
```

Jeg trenger ikke nødvendigvis implementere alt. Poenget er å bruke RNN-en som utgangspunkt for å forstå **hvorfor LSTM, GRU og senere Transformer-arkitekturer ble utviklet**, og samtidig få erfaring med hvordan disse modellene brukes med moderne biblioteker og verktøy.

---

## Sluttmålet

Til slutt vil jeg ha bygget opp en liten språkmodell steg for steg:

```text
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

Det viktigste er ikke å lage en stor eller imponerende LLM. Målet er å bli **familiar med RNN-er og språkmodeller gjennom praktisk arbeid**, forstå de viktigste konseptene underveis, og samtidig bli kjent med verktøyene og bibliotekene som faktisk brukes til å utvikle slike systemer.
