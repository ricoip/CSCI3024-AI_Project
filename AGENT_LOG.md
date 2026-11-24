# Independent checks of the coding agent

Record each check when you do it. For each entry, describe what you asked the
agent for and what it produced, the steps you took to test it independently,
and what you found. If you found a mistake, record the fix and the rule or test
you added. Use your own observations and keep a good result if the check passes.

## Check 1 · Data

Candidate check. Choose rows from `submission/manifest.csv`, find the matching
images in the raw ZIP shards, and compare the square specified by the CSV with
the source and 32×32 preview. Judge the framing, animal, and photo independently
against the handout. The review sheets can help locate pool IDs.

- Context:
- Check:
- Outcome:

## Check 2 · Generation

Candidate check. Run `demo.py` twice using the same label, count, and seed and
compare the pixel digests or the files yourself. Repeat with a different seed.

- Context:
- Check:
- Outcome:

## Check 3 · Results

Candidate check. Open the official evaluation JSON and independently check the
protocol, raw metrics, and hard limits against the handout and report. For a
stronger check, redo the tier calculation from the raw values.

- Context:
- Check:
- Outcome:
