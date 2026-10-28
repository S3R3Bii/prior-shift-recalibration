# Source rights and changes

| Exact benchmark | Creator and original source | Rights boundary | Changes here |
|---|---|---|---|
| OpenML44125 MAGIC Telescope v7 | R. Bock (2004), [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope), DOI10.24432/C52C8B | Original UCI CC BY4.0; OpenML field says Public | Binary labels encoded, exact full-row duplicates removed, whole feature groups selected by hash, maximum8,000 rows |
| OpenML44128 MiniBooNE v6 | Byron Roe (2005), [UCI MiniBooNE particle identification](https://archive.ics.uci.edu/dataset/199/miniboone+particle+identification), DOI10.24432/C5QC87 | Original UCI CC BY4.0; preserve that attribution despite the OpenML CC0 field | Same bounded transformations, measurement descriptors retained, no invented sentinel cleaning |

[CC BY4.0 terms](https://creativecommons.org/licenses/by/4.0/) apply to the source derivatives. No endorsement is implied. These exact OpenML benchmark variants differ in row counts and class balance from their original UCI sources. Public reproducibility does not make the bounded panels representative of either experiment or deployment.

Retrieval URLs, UTC times, exact versions, hashes, selected original row numbers, feature names and duplicate counts are retained in [the manifest](data/manifest.json). Raw ARFF downloads stay outside version control; `python src/fetch.py` retrieves them. No source paper or unpublished manuscript is redistributed. Original code and documentation are MIT licensed separately from benchmark data.
