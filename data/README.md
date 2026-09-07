# data

Corpus, vocabularies and everything the pipeline produces.

## Fetching the vocabularies

```bash
# Run from current directory
./vocabularies/fetch_vocabs.sh          # large; TGN alone is ~40GB
```

One source is not fetchable: the Social History and Industrial Classification is distributed as a PDF only, so `SHIC.txt` beside the script is an extracted copy.

`field_vocab_map.json` records which vocabularies answer which field and is tracked. The seeds under `local/` are review artefacts rather than downloads; they harmonise values taken from every institution, so they stay out of the published tree.

## Fetching Mapping Museums

```bash
# Run from current directory
wget https://museweb.dcs.bbk.ac.uk/static/pdf/MappingMuseumsData2021_09_30.csv -O ./reference/MappingMuseumsData2021_09_30.csv
```

## Gold labelled data

The unlabelled samples for gold labelling sit in `gold/samples/`. Once labelled, they are added to `gold/labels/`, joinable on `id`.

Only the CC0-licensed part of that may be published. The files suffixed `_CC0` hold it, and they are the only gold files git tracks:

```bash
uv run python ../scripts/split_cc0_gold.py --dry-run   # report the counts
uv run python ../scripts/split_cc0_gold.py             # write the _CC0 files
```

Licence is a per-record `ciim/license` unit and is uniform per institution, so an item is CC0 when every record and institution it names is. A findability judgment must also have been made under a query whose seed record is CC0, or the query text would quote a record that is not. The four value-keyed label sets, which name a corpus-wide value rather than a record, are kept where a CC0 institution uses that value. Rerun the split after any redraw or relabel.

