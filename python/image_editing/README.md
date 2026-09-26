# Bildebehandlings-eksperimenter

Dette prosjektet er en liten samling av bildebehandlings-eksperimenter skrevet i Python. Målet er å forstå hvordan bildefiltre fungerer, hvordan pikseldata blir transformert, og hvordan enkle visuelle effekter kan lages ved hjelp av NumPy og Pillow.

## Inkluderte konsepter

- **Grayscale** filter
- **Sepia** effekt
- **Pixelation**
- **Blur** (pågående arbeid)
- **ASCII-style** image rendering
- **Basic image IO and display utilities**

## Verktøy

- Python, NumPy, Pillow


## Eksempel på bruk

Fra `python`-mappen, installer avhengigheter og kjør demoen:

```bash
pip install -r requirements.txt
cd image_editing/filters
python main.py
```

Prosjektet er bygd rundt en gjenbrukbar filter‑pipeline. Hvert filter kan velges og anvendes på et bilde, og koden er utformet for å gjøre eksperimentering enkel.

Konfigurasjon gjøres for øyeblikket i `main.py`, hvor du kan velge hvilket filter som skal brukes og hvilket bilde som skal behandles.

## Hva prosjektet viser

- Bildearrays og pikselmanipulasjon
- Konsepter innen datamaskinsyn (computer vision)
- Python‑biblioteker for praktisk bildebehandling
- Lite, eksperimentelt prosjekt som er enkelt å utvide

## Mulige neste steg

- Legg til flere filtre, for eksempel sharpen, vignette, kontrastjustering og kantdeteksjon
- Legg til et CLI med valgfrie filteralternativer
