---
this_file: README.md
---
# Uubed

Uubed shares embedding inference, compact vector storage and translation-memory
retrieval between book alignment and localization tools.

| Repository | Responsibility |
| --- | --- |
| [uubed-rs](https://github.com/twardoch/uubed-rs) | Rust codecs, portable SQLite translation memory, and the `uubed_native` Python module |
| [uubed-py](https://github.com/twardoch/uubed-py) | Python `uubed` API, model inference, compact records, TMX import and retrieval |
| [uubed-docs](https://github.com/twardoch/uubed-docs) | Documentation site |
| [Vexy Paraltext](https://github.com/vexyart/vexy-paraltext) | Align multilingual books and produce TMX |
| [Abersetz](https://github.com/twardoch/abersetz) | Translate with exact TM matches and similar translation pairs |

For runtime use, install `uubed[tm]` and the desired inference extra. The
`uubed-project` distribution from this repository contains research helpers;
it is not the runtime package. Research chat logs and local data are excluded.

```bash
./test.sh
./build.sh
./publish.sh --dry-run
./publish.sh
```

[RELEASING.md](RELEASING.md) covers credentials, automatic Git semantic tags,
artifact checks and retrying a partial release. The sibling repositories use
the same commands and remain independently buildable. Publish native Uubed,
then Python Uubed, then the consumers. Python versions come from Hatch VCS;
the Rust workspace materializes the release tag in its shared Cargo metadata.

Private corpora, translation memories, model weights, credentials and agent state
belong outside Git. Tests use synthetic examples. MIT license.
