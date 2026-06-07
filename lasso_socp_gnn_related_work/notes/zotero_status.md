# Zotero status

Date: 2026-06-07

## API status

- Zotero was found at `C:\Program Files\Zotero\zotero.exe`.
- Zotero 9.0.4 is running and the local API answers through `curl.exe`.
- PowerShell `Invoke-WebRequest` closes unexpectedly against the Zotero local endpoint, so `curl.exe` was used for probing.

## Existing library

- Existing item count reported by Zotero local API: 24.
- Existing collection count: 1.
- Existing collection name: `10.1007_s11142-023-09811-7-citation`.
- Existing library items are mostly LLM-agent, memory-benchmark, finance, and Scopus snapshot records.
- No obvious title or DOI collision was found against the optimization papers in `bib/references.bib`.

## Import status

The new bibliography has been prepared at:

- `bib/references.bib`
- `bib/zotero_import.bib`

Automatic import was not performed because Zotero's currently selected target is the existing collection `10.1007_s11142-023-09811-7-citation`, and importing there would put these records in the wrong collection.

The local API collection endpoint rejected direct collection creation with:

`Endpoint does not support method`

## Manual completion

Create or select this Zotero collection:

`LASSO-SOCP-GNN Related Work`

Then import:

`C:\Users\34478\Documents\DL-project\papers\lasso_socp_gnn_related_work\bib\zotero_import.bib`

After import, use Zotero's duplicate-items view once. Based on the current library inventory, collisions are unlikely.
