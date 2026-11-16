## Datenquelle
- Datensatz: Züri wie neu 
- Quelle: Stadt Zürich
- URL: https://www.stadt-zuerich.ch/geodaten/download/Zueri_wie_neu
- Lizenz: Creative-Commons-Zero-Lizenz (CC-0)


## Snapshot
- Datei: data/raw/data-zuerich_2026-10-29.csv (nicht im Repo)
- Metadaten: data/raw/metadaten.pdf (vom Portal, nicht im Repo)
- Bereitstellung: CSV und Metadaten als Assets im Release [`data-2026-10-29`](https://github.com/elisab-code/data-analysis-zueri/releases/tag/data-2026-10-29); `python -m src.data.download` lädt die CSV und prüft die Prüfsumme
- Download: 2026-10-29 10:48:05 (MEZ, UTC+1)
- Größe: 61.682.136 Bytes (ca. 61,7 MB)
- SHA-256: 45ee2bc7d5c793d3ee32f2191d86ba8c5aa41551105811b3a432714de6efc5cc
- Umfang: 79.159 Meldungen, 19 Spalten
- Zeitraum (`requested_datetime`): 2013-03-14 15:16:15 bis 2026-10-28 15:25:28

Prüfen: `sha256sum data/raw/data-zuerich_2026-10-29.csv`

## Format
- CSV, Trennzeichen Komma, UTF-8 mit BOM (einlesen mit `encoding="utf-8-sig"`)
- Die Datei nicht in Excel öffnen und speichern: Das verändert Trennzeichen, Kodierung und Prüfsumme.
- Bekannter Mangel der Quelle: Textfelder enthalten Mojibake (z. B. `Ã¼` statt `ü`), siehe `notebooks/01_data-review.ipynb`.
