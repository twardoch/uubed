---
this_file: WORK.md
---

## Release and privacy cleanup

Implemented standalone tag-derived release commands, source/archive privacy gates, fresh artifact manifests, dry runs and same-tag retries. Verification is recorded in the workspace release report. Existing local data remains on disk.

### Release verification — 2026-09-12

49 tests passed, 2 skipped; v1.0.7 wheel/sdist preview verified. Shared release-flow suite: 11 passed. Source HEAD and latest tag remained unchanged. No live publication was performed.

## 2026-09-12 — Cargo release packaging regression

The live v1.0.12 attempt exposed two gaps: Cargo's broad include globs admitted an ignored examples/.DS_Store, and cargo metadata --no-deps did not update workspace versions in Cargo.lock before tagging. Narrowed all three crate source/test/benchmark/example includes to Rust files; version synchronization now resolves full metadata before committing, and crate packaging requires --locked. The archive rejection guard remains in place.

Regression checks reproduced both failures before the fix. After the fix, all 12 shared release tests and the three-crate packaging canary test passed. Built the native wheel, source distribution, and both Cargo archives directly from the real working tree with the original .DS_Store present; archive checks and compilation of both packaged crates passed. No runtime Rust code changed.

The failed v1.0.12 tag remains local and unchanged; the remote has no such tag. No live push/upload was performed during repair. Because the fix changes source, rerun the normal ./publish.sh command to create the next patch version (v1.0.13); --resume v1.0.12 is inappropriate for this modified checkout. Verification artifacts are outside publish destinations in ignored research/releases/packaging-fix-h0cmw4d_/artifacts.

## 2026-09-12 — License-file release validation

The shared archive gate now checks every License-File declaration in metadata
2.4+ against a regular file at the required sdist or wheel location. Missing,
misplaced and directory-only licenses fail before pushing or uploading. Synced
the helper across all six repositories. Verification: 14 release-tool tests
passed; nine existing sibling preview artifacts passed the stronger checks.
