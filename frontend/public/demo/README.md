# Demo images (issue #122)

Pair for one-click "Load demo": same identity, different session/lighting.

| Slot | File | Source | License |
|---|---|---|---|
| A | `obama-a.jpg` | `President Barack Obama (square).jpg`, official Oval Office portrait Dec 6 2012 by Pete Souza, via Wikimedia Commons `Special:FilePath/President Barack Obama (square).jpg` | Public domain (work of the U.S. federal government) |
| B | `obama-b.jpg` | `Official portrait of Barack Obama.jpg`, official portrait Jan 13 2009 by Pete Souza, via Wikimedia Commons `Special:FilePath/Official portrait of Barack Obama.jpg` | CC-BY 3.0, attribution: Pete Souza / Obama-Biden Transition Project |

Normalization: center-crop to square + Lanczos resize to 1024x1024, JPEG q92.

- `obama-a.jpg`: 1024x1024, 237343 bytes, sha256 `0bf056708b516553b03b5ea29e4ce225270ef90736f00c088783d2ca5fd61812`
- `obama-b.jpg`: 1024x1024, 184716 bytes, sha256 `df05b7fb2dca973d15be6b09a12c37654c96b59f538f310702040d35c5053cae`

Both pass `ImageBytes::parse` (magic `FF D8 FF`, well under `MAX_IMAGE_BYTES` 8 MiB).
Photo credit for slot B must stay visible wherever the demo is labeled.
