# data/

Keep external or generated data out of git unless it is tiny and required for tests.

Recommended subfolders:

```text
data/raw/          # external/reference data, never modified by scripts
data/processed/    # processed arrays generated from raw data
data/metadata/     # source notes, digitization metadata, etc.
```
