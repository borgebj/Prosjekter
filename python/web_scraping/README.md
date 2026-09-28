
# Web Scraping

Dette prosjektet er en liten samling web scraping-eksperimenter skrevet i Python. Målet er å hente, analysere og behandle data fra nettsider.
Koden er hovdsakelig hentet fra et studieprosjekt fra emnet IN4110, med fokus på Python for dypere problemløsing.

## Inkluderte konsepter

* **Web scraping og HTML-parsing**
* **URL- og datoekstraksjon**
* **Databehandling og visualisering**
* **BFS på Wikipedia**

## Verktøy

* Python, Requests, BeautifulSoup, Pandas, Matplotlib, Tabulate

## Eksempel på bruk

Installer verktøy:

```bash
pip install -r python/web_scraping/requirements.txt
```

Eksemplene kan deretter kjøres individuelt fra `python/web_scraping/src/`, for eksempel:

```bash
python python/web_scraping/src/time_planner.py
```
Henter en alpinkalender fra Wikipedia og lager en strukturert oversikt over arrangementene.  
```bash
python python/web_scraping/src/fetch_player_statistics.py
```
Henter NBA-statistikk fra Wikipedia og genererer visualiseringer av poeng, assists og rebounds.
```bash
python python/web_scraping/src/wiki_race_challenge.py
```
Kjører et morsomt program: raskeste vei fra en link til en annen


## Hva prosjektet viser

* Henting og behandling av data fra nettsider
* Parsing av HTML og Wikipedia-tabeller
* Databehandling med Pandas
* Visualisering av innsamlede data
* BFS for å finne korteste vei mellom Wikipedia-artikler
