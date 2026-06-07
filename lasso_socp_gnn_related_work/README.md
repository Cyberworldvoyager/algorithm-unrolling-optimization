# LASSO-SOCP-GNN related work package

This folder contains a literature package for the proposed LASSO-GNN and SOCP-GNN learned optimization project.

## Contents

- `pdfs/`: downloaded or copied PDFs for the cited papers.
- `bib/references.bib`: BibTeX database for the LaTeX draft and Zotero import.
- `latex/related_work.tex`: compilable conference-style related work draft. The source intentionally avoids conference-template branding.
- `notes/novelty_audit.md`: evidence-based novelty positioning and title recommendations.

## Zotero import

Recommended collection name:

`LASSO-SOCP-GNN Related Work`

Zotero is running and the local API is readable through `curl.exe`. Existing items were checked, and no obvious title or DOI collision was found. Automatic import was not performed because Zotero's currently selected target is an unrelated existing collection, and the local API did not allow direct creation of a new collection. Import `bib/zotero_import.bib` into the collection above after creating or selecting it in Zotero.

## Build

From `latex/`:

```powershell
pdflatex related_work.tex
bibtex related_work
pdflatex related_work.tex
pdflatex related_work.tex
```

If `pdflatex` is not installed locally, the `.tex` and `.bib` files can still be uploaded to Overleaf or compiled in any standard LaTeX environment.
