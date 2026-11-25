# Contributing to Pronunciada

Bug reports, new languages, scoring changes and documentation fixes are welcome. Any change to
scoring has to be evaluated on data it was not tuned on. The procedure is described below.

## Setup

```bash
git clone https://github.com/moglearn/pronunciada
cd pronunciada
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
pytest -q
```

The first run downloads the acoustic model (about 250 MB) to `~/.cache/pronunciada`.

## Types of contribution

**Bug reports.** Include the language, the reference text, the output and the output you expected.
Attach a short recording if you are allowed to share one.

**A new language.** You need a text front-end that turns text into expected sounds
(`pronunciada/text/`), a small output adapter trained on native speech, and a calibration
offset. `docs/DESIGN.md` describes each part. Open an issue first to agree on the
phoneme inventory.

**Scoring changes.** Follow the procedure used for every round under `results/`:

1. Before measuring, state what you expect to improve and the threshold it has to meet.
2. Select on development data (training splits, FLEURS validation, the dev halves).
3. Run the held-out benchmarks once and report the results.
4. Check that scores for native speakers do not drop (the guards in `bench/calib/`).

A change that misses its threshold can still be written up. `results/` contains several.

**Documentation, examples and integrations.** Examples must run as written.

## Pull requests

- One change per pull request.
- `pytest -q` must pass. Scores for existing languages must stay byte-identical unless the
  pull request is a scoring change with the measurements above.
- Do not add recordings or third-party data to the repository. Point to the source and its
  license instead, and add it to `docs/LICENSING.md`.
- Avoid new runtime dependencies. The library depends on numpy, scipy, ONNX Runtime, kaldi-native-fbank and soundfile.

## License

By contributing you agree that your contribution is licensed under Apache-2.0.
