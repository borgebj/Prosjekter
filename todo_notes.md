# TODO Notes

Ideer til prosjekter jeg har lyst til å jobbe med senere.

Prosjektene er sortert etter anbefalt rekkefølge. Eksisterende prosjekter kommer først, siden disse hovedsakelig skal hentes frem, gjøres kjørbare og eventuelt videreutvikles. Nye prosjekter følger deretter etter anbefalt læringskurve.

Tidene er omtrentlige og avhenger av hvor omfattende prosjektet gjøres.

*(Forslag fra AI med tanke på interesser, tidligere erfaring og mulig relevans for jobb.)*

---

# Eksisterende prosjekter

## 1. Web scraping / datapipeline

**Utgangspunkt:** IN4110 – Problemløsning med høynivå-språk, Assignment 4
**Vanskelighetsgrad:** 🟢 Lett
**Estimert tid:** 1–3 dager

Et eksisterende web scraping-prosjekt som først og fremst trenger å hentes frem og fås til å fungere igjen.

Målet er ikke nødvendigvis å bygge det om fra bunnen, men å få oversikt over gammel kode og gjøre prosjektet presentabelt.

### Fase 1 – Få prosjektet til å fungere

* [ ] Hente frem prosjektet
* [ ] Undersøke hvilke dependencies som brukes
* [ ] Få koden til å kjøre igjen
* [ ] Fikse eventuelle problemer med gamle biblioteker / API-er
* [ ] Forstå hva koden gjør

### Fase 2 – Rydde opp

* [ ] Rydde opp i prosjektstrukturen
* [ ] Fjerne unødvendig kode
* [ ] Forbedre variabel- og funksjonsnavn der det er nødvendig
* [ ] Legge til / oppdatere requirements
* [ ] Sørge for at prosjektet er enkelt å kjøre

### Fase 3 – Gjøre prosjektet presentabelt

* [ ] Lage README
* [ ] Dokumentere hva som scrapes
* [ ] Dokumentere hvordan prosjektet kjøres
* [ ] Vise eksempel på resultat
* [ ] Eventuelt legge til visualisering av data

### Eventuelt senere

* [ ] Gjøre scraping mer robust
* [ ] Automatisere kjøring
* [ ] Lagre data i et mer passende format / database
* [ ] Lage en komplett scraping → processing → analysis pipeline
* [ ] Eventuelt bruke dataene i et ML-prosjekt

**Mål:** Få et tidligere universitetsprosjekt tilbake i fungerende og presentabel stand, samtidig som jeg får frisket opp erfaring med web scraping og data processing.

---

## 2. Distributed System

**Utgangspunkt:** IN5020 – Distribuerte systemer
**Vanskelighetsgrad:** 🟠 Middels
**Estimert tid:** 3–7 dager

Ta utgangspunkt i de eksisterende oppgavene under `ass1`, `ass2` og `ass3`.

Målet er først å finne ut hva som faktisk er interessant å ta vare på, i stedet for å starte et helt nytt distributed systems-prosjekt.

### Fase 1 – Hente frem prosjektet

* [ ] Gå gjennom `ass1`
* [ ] Gå gjennom `ass2`
* [ ] Gå gjennom `ass3`
* [ ] Forstå hva de forskjellige oppgavene gjør
* [ ] Finne ut hvilke dependencies / verktøy som kreves
* [ ] Få valgt prosjekt til å kjøre igjen
* [ ] Kjøre eksisterende tester

### Fase 2 – Velge prosjekt

* [ ] Velge den mest interessante oppgaven
* [ ] Vurdere om deler fra flere oppgaver kan kombineres
* [ ] Forstå arkitekturen
* [ ] Dokumentere hvordan systemet fungerer

### Fase 3 – Gjøre det presentabelt

* [ ] Rydde opp i koden
* [ ] Oppdatere README
* [ ] Dokumentere hvordan systemet kjøres
* [ ] Dokumentere arkitekturen
* [ ] Lage en enkel demo
* [ ] Eventuelt legge til diagram over systemet

### Eventuelt videreutvikle

* [ ] Videreutvikle replication / peer-to-peer-delen
* [ ] Teste systemet med flere noder
* [ ] Teste hva som skjer når noder feiler
* [ ] Benchmarke systemet
* [ ] Forbedre feilhåndtering

**Mål:** Gjøre et tidligere universitetsprosjekt om til et ryddig og demonstrerbart GitHub-prosjekt.

---

# Nye prosjekter

## 3. Data Processing + Anomaly Detection

**Vanskelighetsgrad:** 🟢 Middels
**Estimert tid:** 2–3 uker

Et prosjekt med fokus på praktisk data processing, data quality, analyse, feature engineering og anomaly detection.

Målet er å starte med rådata og bygge en liten, realistisk pipeline som først forstår og kvalitetssikrer dataene, før forskjellige metoder for anomaly detection testes.

Tidsseriedata eller industrielle sensordata er et mulig utgangspunkt, men datasettet velges etter hva som gir et interessant problem å undersøke.

```text
Raw Data
   ↓
Data Inspection
   ↓
Data Cleaning & Validation
   ↓
Data Analysis
   ↓
Feature Engineering
   ↓
Anomaly Detection
   ↓
Evaluation
   ↓
Visualization / Report
```

### V1 – Data processing & data quality

* [ ] Finne et realistisk datasett
* [ ] Forstå hva datasettet representerer
* [ ] Lese inn og strukturere data
* [ ] Analysere datatyper og kolonner
* [ ] Undersøke missing values
* [ ] Finne og håndtere duplikater
* [ ] Validere verdier
* [ ] Håndtere timestamps dersom relevant
* [ ] Undersøke ugyldige eller urealistiske observasjoner
* [ ] Analysere statistikk og distribusjoner
* [ ] Visualisere data
* [ ] Dokumentere problemer som finnes i rådataene

### V2 – Statistisk anomaly detection

* [ ] Definere hva som skal regnes som en anomali
* [ ] Implementere enkle thresholds
* [ ] Rolling mean
* [ ] Rolling standard deviation
* [ ] Z-score
* [ ] IQR
* [ ] Identifisere outliers
* [ ] Visualisere anomalies i tidsserier
* [ ] Undersøke false positives
* [ ] Sammenligne forskjellige statistiske metoder

### V3 – Machine learning

* [ ] Lage relevante features
* [ ] Prøve Isolation Forest
* [ ] Eventuelt Local Outlier Factor
* [ ] Eventuelt DBSCAN
* [ ] Sammenligne forskjellige metoder
* [ ] Evaluere resultatene
* [ ] Undersøke hvilke features som påvirker resultatet
* [ ] Sammenligne statistiske metoder med ML-metoder

### V4 – Pipeline

* [ ] Lage en gjenbrukbar data pipeline
* [ ] Automatisk kjøre analysen på nye data
* [ ] Separere data processing, feature engineering og anomaly detection
* [ ] Generere en enkel rapport
* [ ] Visualisere oppdagede anomalies
* [ ] Eventuelt simulere nye data
* [ ] Eventuelt lage et enkelt dashboard

### Ekstra – Lag egne anomalies

Dersom datasettet mangler tydelig merkede anomalies, kan kunstige problemer introduseres i et kontrollert datasett.

Eksempler:

* [ ] Missing values
* [ ] Ekstreme verdier
* [ ] Plutselige spikes
* [ ] Gradvis sensor drift
* [ ] Sensor som blir stående på samme verdi
* [ ] Urealistiske målinger

Dette gjør det mulig å undersøke hvor godt metodene faktisk klarer å oppdage kjente anomalies.

**Eksempel:**

```text
Sensor data
     ↓
Temperature
Pressure
Vibration
Current
     ↓
Data validation
     ↓
Feature engineering
     ↓
Anomaly detection
     ↓
Potential anomaly
```

**Mål:** Få praktisk erfaring med data processing, data quality, feature engineering, statistisk analyse og anomaly detection, samtidig som jeg lærer å bygge en liten og gjenbrukbar data pipeline.


## 4. CNN + MNIST

**Vanskelighetsgrad:** 🟢 Lett–middels
**Estimert tid:** 1–2 uker

Et naturlig første steg videre fra grunnleggende maskinlæring og nevrale nettverk.

* [ ] Laste inn MNIST
* [ ] Lage en enkel neural network-baseline
* [ ] Implementere convolution
* [ ] Implementere ReLU
* [ ] Implementere pooling
* [ ] Lage en enkel CNN
* [ ] Trene modellen
* [ ] Evaluere accuracy
* [ ] Visualisere feilklassifiseringer
* [ ] Undersøke hvilke typer bilder modellen gjør feil på
* [ ] Eventuelt sammenligne egen implementasjon med PyTorch

**Mål:** Forstå hvordan CNN-er fungerer i praksis, ikke bare bruke et ferdig bibliotek.

---

## 5. Embeddings + Vector Search

**Vanskelighetsgrad:** 🟢 Middels
**Estimert tid:** 1–2 uker

Lære hvordan tekst kan representeres som vektorer og brukes til semantic search.

* [ ] Gjøre tekst om til embeddings
* [ ] Forstå hva embeddings representerer
* [ ] Implementere cosine similarity
* [ ] Søke etter semantisk lignende tekst
* [ ] Lage et lite datasett
* [ ] Evaluere hvor godt søket fungerer
* [ ] Visualisere embedding space
* [ ] Eksperimentere med PCA / t-SNE / UMAP
* [ ] Sammenligne forskjellige embedding-modeller
* [ ] Eventuelt teste en vector database

**Eksempel:**

```text
"How do I blur an image?"
              ↓
         embedding
              ↓
       vector search
              ↓
"Where is Gaussian blur implemented?"
```

**Mål:** Forstå embeddings og semantic search før jeg begynner med RAG og LLM-er.

---

## 6. Transformer fra scratch

**Vanskelighetsgrad:** 🟠 Middels–vanskelig
**Estimert tid:** 2–4 uker

Lage en liten Transformer for å forstå hvordan moderne språkmodeller fungerer.

Ikke målet å lage en ChatGPT-lignende modell. Målet er å forstå arkitekturen.

* [ ] Tokenisering
* [ ] Embeddings
* [ ] Positional encoding
* [ ] Self-attention
* [ ] Multi-head attention
* [ ] Feed-forward network
* [ ] Transformer block
* [ ] Output layer
* [ ] Trene en veldig liten modell på et enkelt datasett
* [ ] Eksperimentere med forskjellige størrelser
* [ ] Visualisere attention weights hvis mulig
* [ ] Dokumentere hvordan de forskjellige delene fungerer

```text
Tokens
 ↓
Embeddings
 ↓
Positional Encoding
 ↓
Self-Attention
 ↓
Multi-Head Attention
 ↓
Feed-Forward
 ↓
Transformer Block
 ↓
Output
```

**Mål:** Forstå Transformer-arkitekturen gjennom egen implementasjon.

---

## 7. Lokal kodeassistent

**Vanskelighetsgrad:** 🟠 Middels–vanskelig
**Estimert tid:** 2–4 uker

Bygge en liten assistent som kan søke gjennom egne kodeprosjekter.

Dette skal **ikke** starte med å lage eller trene en egen språkmodell.

### V1 – Semantic search

* [ ] Lese inn kodefiler
* [ ] Dele kode opp i passende chunks
* [ ] Lage embeddings
* [ ] Bruke cosine similarity / vector search
* [ ] Søke etter relevant kode
* [ ] Returnere filnavn og linjenummer

```text
Kodebase
   ↓
Embeddings
   ↓
Vector Search
   ↓
Relevant kode
```

### V2 – Bedre kodeforståelse

* [ ] Eksperimentere med forskjellige chunking-strategier
* [ ] Skille mellom funksjoner, klasser og hele filer
* [ ] Eventuelt bruke AST til å analysere Python-kode
* [ ] Søke både i kode og dokumentasjon
* [ ] Evaluere hvilke chunks som faktisk er relevante

### V3 – Visualisering

* [ ] Visualisere kode-embeddings
* [ ] PCA / UMAP
* [ ] Undersøke hvordan ulike typer kode grupperer seg

### V4 – LLM / RAG

* [ ] Koble til en ferdig LLM
* [ ] Sende spørsmålet + relevante kodebiter til modellen
* [ ] Generere svar
* [ ] Legge til kildehenvisninger til filer
* [ ] Svare på spørsmål om kodebasen

```text
Spørsmål
   ↓
Embedding
   ↓
Vector Search
   ↓
Relevant kode
   ↓
LLM
   ↓
Svar
```

### V5 – Eventuelt lokal LLM

* [ ] Kjøre en ferdigtrent LLM lokalt
* [ ] Integrere den med retrieval-systemet
* [ ] Eksperimentere med forskjellige modeller
* [ ] Sammenligne lokal modell med API-basert modell

**Mål:** Lære hvordan embeddings, retrieval og LLM-er kan kombineres uten å måtte trene en språkmodell selv.

---

## 8. PDF + RAG

**Vanskelighetsgrad:** 🟠 Middels–vanskelig
**Estimert tid:** 2–4 uker

Bygge en liten applikasjon som kan stille spørsmål til PDF-er og dokumenter.

```text
PDF
 ↓
Tekst
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector Search
 ↓
Relevante chunks
 ↓
LLM
 ↓
Svar + kilder
```

* [ ] Laste inn PDF
* [ ] Ekstrahere tekst
* [ ] Håndtere dokumenter med flere sider
* [ ] Dele dokumentet opp i chunks
* [ ] Eksperimentere med chunk size
* [ ] Lage embeddings
* [ ] Søke etter relevante chunks
* [ ] Sende relevante chunks til en ferdig LLM
* [ ] Generere svar
* [ ] Vise hvilke deler av dokumentet svaret bygger på
* [ ] Eventuelt støtte flere dokumenter
* [ ] Eventuelt sammenligne forskjellige retrieval-metoder

**Mål:** Forstå og implementere en enkel RAG-pipeline uten å trene en egen LLM.

---

# Senere / større prosjekter

## 9. Message Queue

**Vanskelighetsgrad:** 🔴 Vanskelig
**Estimert tid:** 2–4 uker

Ta eventuelt utgangspunkt i tidligere arbeid fra IN2140.

Grunnleggende arkitektur:

```text
Producer
   ↓
 Queue
   ↓
Consumer
```

Videreutvikling:

* [ ] Producer
* [ ] Queue
* [ ] Consumer
* [ ] Flere consumers
* [ ] Acknowledgements
* [ ] Retries
* [ ] Feilhåndtering
* [ ] Persistence
* [ ] Concurrency
* [ ] Eventuelt prioritetsbaserte meldinger
* [ ] Eventuelt benchmarke systemet
* [ ] Undersøke hva som skjer ved consumer/server-crash

**Mål:** Lage et lite, men realistisk meldingssystem og forstå hvordan slike systemer fungerer.

---

# Prosjektkombinasjoner

Noen av prosjektene kan kombineres i stedet for å bli separate prosjekter.

### Web scraping → Data Processing → Anomaly Detection

De to første prosjektene kan potensielt kobles sammen dersom datasettet fra scraping-prosjektet passer.

```text
Web scraping
     ↓
Data processing
     ↓
Feature engineering
     ↓
ML / anomaly detection
```

Dette kan bli en komplett data pipeline fra innsamling av rådata til analyse og ML.

### Embeddings → Lokal kodeassistent

Embeddings-prosjektet kan fungere som første del av kodeassistenten.

```text
Embeddings
     ↓
Vector Search
     ↓
Kodeassistent
     ↓
RAG
     ↓
LLM
```

Da slipper jeg å bygge samme teknologi to ganger.

### Transformer → LLM/RAG

Transformer-prosjektet kan gi forståelse av hva som skjer inne i språkmodellene, mens kodeassistent/PDF-RAG viser hvordan ferdigtrente modeller kan brukes i praktiske systemer.

### Distributed System → Message Queue

Distributed System-prosjektet kan gi et naturlig utgangspunkt for senere arbeid med message queues, concurrency, fault tolerance og kommunikasjon mellom noder.

---