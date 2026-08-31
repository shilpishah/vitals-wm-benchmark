"""Real generative video model backend names -- the SINGLE canonical list
(AGENT.md M6, 2026-08, found via a real bug: `render_survival_report.py`'s
own color/category logic hardcoded `backend == "cosmos"` as the only
non-Truth, non-Baseline category, so Wan2.1's own results silently got
miscategorized as a Baseline -- a real predictor gets grouped with
ConstantVelocity/CopyLastState, not with Cosmos, purely because a string
comparison only knew about one model's name).

Both `scripts/run_model_population.py` (which backend names it accepts
via `--model`) and `scripts/render_survival_report.py` (which category/
color a results file gets) import this SAME set -- adding a new model
means adding its name here ONCE, not remembering to update two files that
would otherwise silently drift apart exactly like this bug did.

`cosmos14b` (2026-08, requested directly: "doesn't that mean something
with the benchmark's parameters are off" -- the 2B checkpoint was always
a deliberate cost/turnaround choice for the FIRST run, not the largest
Cosmos-Predict2 variant, see cosmos.py's own docstring) is its OWN backend
name, not a `--cosmos-model-size` flag on the existing `cosmos` entry --
a separate, comparable results file (`l0_demo_<scenario>_cosmos14b.json`)
that shows up as its own line/page everywhere `cosmos`/`wan` already do,
rather than silently overwriting the 2B result under the same name.

`cosmos720p` (2026-08, same conversation, the SAME "what parameters
might be off" question applied to a different honest caveat --
`COSMOS_RESOLUTION` was chosen "for cost/turnaround, not quality," never
tested at the higher setting Cosmos also supports). Isolated as its own
backend, at the DEFAULT model_size ("2B"), specifically so it varies
resolution alone against the SAME 2B baseline the 14B test already used
-- one axis at a time (this project's own established calibration
discipline), not combined with the model-size question into one
confounded experiment."""
REAL_MODEL_BACKENDS = {"cosmos", "cosmos14b", "cosmos720p", "wan"}
