# Release and DOI Archive Runbook

This repository is not submission ready until a DOI-backed software archive exists. The current local release candidate is `0.2.2`; `ARCHIVE_MANIFEST.json` records the release-critical files and hashes.

## Preconditions

- The worktree changes for the release candidate are committed.
- `paper/main.pdf` and `paper/supplement/supplement.pdf` were rebuilt from the current `reports/paper` assets after the last asset regeneration, so the tracked PDFs match a clean `make paper-check` rebuild. Never tag a release whose PDFs predate the assets they typeset.
- `poetry check --strict`, `poetry run mypy src`, `poetry run ruff check src tests`, and `poetry run pytest` pass.
- `poetry build` creates `dist/dyn_evt_pdm-0.2.2.tar.gz` and `dist/dyn_evt_pdm-0.2.2-py3-none-any.whl`.
- `poetry run dyn-evt build-submission-package --paper-root paper --asset-root reports/paper --output-root reports/submission` reports only the DOI archive blocker.

## GitHub Release

After committing the release-candidate state:

```powershell
git tag -a v0.2.2 -m "Release v0.2.2"
git push origin main
git push origin v0.2.2
```

Create a GitHub release from tag `v0.2.2` and attach:

- `dist/dyn_evt_pdm-0.2.2.tar.gz`
- `dist/dyn_evt_pdm-0.2.2-py3-none-any.whl`
- `paper/main.pdf`
- `paper/supplement/supplement.pdf`
- `ARCHIVE_MANIFEST.json`
- `reports/submission/submission_manifest.json`

## DOI Archive

Create a Zenodo or equivalent DOI-backed archive from the GitHub release. Do not edit DOI fields until the archive provider returns a real DOI.

The `v0.2.2` tag includes `.zenodo.json` in the tagged tree. If using `v0.2.0` instead, enter the metadata from `.zenodo.json` manually in the archive provider because that earlier tag did not contain the file.

After the DOI exists:

1. Add the DOI to `CITATION.cff`.
2. Add an identifier entry to `codemeta.json`.
3. Update `ARCHIVE_MANIFEST.json`:
   - `doi_status`
   - release commit hash
   - DOI string
4. Update the paper availability text and bibliography entry for the software release.
5. Rebuild the submission package.

The final submission decision may only change after the DOI-backed archive is present and cited.
