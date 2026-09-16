"""Real generative video model backend registry -- the SINGLE canonical
source (AGENT.md M6, 2026-08, found via a real bug: `render_survival_
report.py`'s own color/category logic hardcoded `backend == "cosmos"` as
the only non-Truth, non-Baseline category, so Wan2.1's own results
silently got miscategorized as a Baseline -- a real predictor gets
grouped with ConstantVelocity/CopyLastState, not with Cosmos, purely
because a string comparison only knew about one model's name).

Both `scripts/run_model_population.py` (which backend names it accepts
via `--model`) and `scripts/render_survival_report.py` (which category/
color a results file gets) import `REAL_MODEL_BACKENDS` -- adding a new
UNRESTRICTED model means adding one entry to `MODEL_REGISTRY` below, not
remembering to update two files that would otherwise silently drift apart
exactly like this bug did.

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
confounded experiment.

## Restricted-license models (2026-09)

VITALS is meant to be handed to other companies to evaluate their own
models (see ADDING_A_MODEL.md), which means a model with a jurisdiction-
or scale-restricted license cannot simply be added the same casual way
`cosmos14b`/`cosmos720p` were -- some adopters would be handed a
component they are not legally permitted to use, with nothing in the
benchmark itself flagging that. `MODEL_REGISTRY` carries a `restricted`
flag per model instead of silently including everything in
`REAL_MODEL_BACKENDS`: a restricted model never appears in `--model`
choices, a report's model list, or an auto-detected lambda-sweep
condition unless a human has explicitly opted in for THIS invocation via
`VITALS_ENABLE_RESTRICTED_MODELS` -- never on by default, and never
persisted anywhere that would make it easy to forget it's on.

This is deliberately NOT a second registry a new model could be added to
instead of `MODEL_REGISTRY` -- that would reintroduce exactly the
two-places-drift bug documented above, just for restricted models
specifically. There remains exactly one place that answers "which models
exist"; restricted entries are just a subset of it, filtered at import
time by an explicit opt-in rather than always active.

When adding a restricted model: register it here with its real license
name and a one-line `restriction_note` a human can act on (what's
actually restricted, not just "has a restrictive license"), and state the
same restriction plainly in the model's own adapter module docstring --
the same honesty `wan.py`'s own docstring already models for ITS license
tradeoffs (Apache 2.0, not gated, chosen partly because of that).
"""
import os

MODEL_REGISTRY = {
    # `superseded_by` (2026-09): the newer release of the SAME model family.
    # Consumed by scripts/render_score_report.py, which drops a superseded
    # model's page for a scenario ONLY once its successor has a real
    # results file for that same scenario (per-scenario, never globally --
    # a superseded page is never dropped where the successor has not yet
    # been run, or a scenario would silently lose its only real-model
    # result). The old results files are never deleted; only the report's
    # default view changes, and the cover page lists what it omitted.
    "cosmos":     dict(license="nvidia-open-model-license", restricted=False, superseded_by="cosmos3nano"),
    "cosmos14b":  dict(license="nvidia-open-model-license", restricted=False, superseded_by="cosmos3nano"),
    "cosmos720p": dict(license="nvidia-open-model-license", restricted=False, superseded_by="cosmos3nano"),
    "wan":        dict(license="apache-2.0", restricted=False, superseded_by="wan22"),
    # NVIDIA Cosmos 3 Nano (2026-09): OpenMDW-1.1 per the model card, not
    # gated, commercial OK -> unrestricted. Nano only -- Super needs
    # multi-GPU tensor parallelism (see remote/modal_app_cosmos3.py).
    "cosmos3nano": dict(license="openmdw-1.1", restricted=False),
    # Cosmos 3 SUPER (2026-09-12, user: "loop in real cosmos 3 instead of
    # just cosmos 3 nano"): the full ~132 GB model, `nvidia/Cosmos3-Super`
    # (public, OpenMDW-1.1), served by remote/modal_app_cosmos3super.py
    # under tensor parallelism on 4x H100 -- see that file for the
    # documented reasons it cannot run the way Nano does. Nano stays
    # registered alongside it (both were asked for); no supersession.
    "cosmos3super": dict(license="openmdw-1.1", restricted=False),
    # Wan2.2 I2V-A14B (2026-09): Apache 2.0, not gated. The open successor to
    # `wan`; its own name so populations never overwrite 2.1's.
    "wan22":       dict(license="apache-2.0", restricted=False),
    # LTX-2.3 (2026-09): VERIFIED against the LTX-2.x Community License text
    # (github.com/Lightricks/LTX-2, LICENSE-2_x): entities with annual
    # revenue >= $10,000,000 must obtain a paid license (Sec. 2.1); OFAC/
    # export restrictions (Sec. 7); prohibited uses include competing with
    # Lightricks' products and training competing models. Same clause class
    # as hunyuan/cogvideox15 -> restricted, opt-in per run.
    "ltx23":       dict(license="ltx-2.x-community-license", restricted=True,
                        restriction_note="LTX-2.x Community License: paid license required for entities with "
                                         "annual revenue >= $10,000,000 (Sec. 2.1); export/OFAC restrictions; "
                                         "no competing products/models -- see remote/modal_app_ltx.py"),
    # CogVideoX1.5-5B-I2V (2026-09): the "CogVideoX LICENSE" requires
    # registration for commercial use, caps it at 1M visits/month, and is
    # governed by PRC law -- the same class of clause that made hunyuan
    # restricted. Opt in per run, never on by default.
    "cogvideox15": dict(license="cogvideox-license", restricted=True,
                        restriction_note="Commercial use requires registering for a license with Zhipu/THUDM "
                                         "and is capped at 1M visits/month; PRC governing law; see "
                                         "remote/modal_app_cogvideox.py's own module docstring"),
    "hunyuan":    dict(license="tencent-hunyuan-community", restricted=True,
                       restriction_note="Not licensed for use in the EU, UK, or South Korea; "
                                         "see vitals/adapters/hunyuan.py's own module docstring"),
    # API-access models (2026-09) -- `access="api"`: no open weights, no
    # Modal deployment, inference happens on the provider's hosted
    # endpoint (vitals/adapters/runway.py). Deliberately NOT `restricted`:
    # that flag means "license needs a per-run opt-in"; an API model's
    # gate is a key (RUNWAYML_API_SECRET) and per-call billing, which the
    # adapter enforces itself. `license` here is the provider's terms of
    # service, not a weights license -- a different kind of artifact, named
    # as such. One backend NAME per gateway model id, same rule as
    # cosmos14b/cosmos720p: results must never overwrite each other.
    # Runway gateway models -- ONE flagship per family (user, 2026-09-12:
    # "i don't need all of the different versions of the models in runway,
    # i just need the highest ones"): the turbo/fast/standard/mini tiers
    # are deliberately not registered. Same adapter, same frozen prompt,
    # same single-image conditioning; only the gateway model id, its
    # allowed aspect ratio and its allowed clip durations differ. `ratio`
    # and `durations` are the values documented for each model at the
    # time of writing and are NOT yet verified by a real call except for
    # gen4.5 (2s clips accepted, 24 credits) -- the first call of each
    # other model is its debugging pass (the gateway's error names the
    # allowed values). A model whose minimum clip exceeds the 2s this
    # protocol needs is billed for that minimum and trimmed: veo/kling/
    # hailuo-class calls cost several times a gen4.5 call. All
    # `restricted=False`: the gate is the key + per-call billing.
    "runway_gen4.5":       dict(license="runway-api-terms", restricted=False, access="api",
                                provider="runway", model_id="gen4.5", ratio="1104:832"),
    # The gateway flagships the user chose (2026-09-14: "veo3.1, seedance
    # 2.5, gemini omni (whatever the flagship is), and minimax h3 max";
    # explicitly NOT wan3, and never kling -- 2026-09-13). Ids are the
    # tier's own (organization.retrieve()): veo3.1 (never veo3),
    # seedance2_5 (not seedance2), gemini_omni_flash_1.1 (the newer of
    # the two gemini_omni entries), h3_max (MiniMax Hailuo 3 Max).
    "runway_veo3.1":       dict(license="runway-api-terms", restricted=False, access="api",
                                provider="runway", model_id="veo3.1", ratio="1280:720", durations=(4, 6, 8)),
    "runway_seedance2.5":  dict(license="runway-api-terms", restricted=False, access="api",
                                provider="runway", model_id="seedance2_5", ratio="1280:720", durations=(5, 10)),
    # gemini_omni rejects `seed` (not seed-reproducible provider-side);
    # h3_max rejects `ratio` (ratio=None omits it) -- both from the
    # 2026-09-14 smokes' own 400 responses.
    # `seed_reproducible=False` (2026-09-14): the gateway accepts no seed
    # for this model, so re-running its population does NOT reproduce the
    # same videos. The cross-model comparison is unaffected -- pairing in
    # VITALS is on the EPISODE (the same rendered prefix and the same
    # perturbed initial condition per seed for every model, which this
    # project controls), not on the provider's sampler -- but this one
    # population is not bit-replicable, and the report says so per model.
    "runway_gemini_omni":  dict(license="runway-api-terms", restricted=False, access="api",
                                provider="runway", model_id="gemini_omni_flash_1.1", ratio="1280:720", durations=(4, 6, 8),
                                supports_seed=False, seed_reproducible=False),
    "runway_h3_max":       dict(license="runway-api-terms", restricted=False, access="api",
                                provider="runway", model_id="h3_max", ratio=None, durations=(6, 10)),
}


def _enabled_restricted_models():
    """Explicit, per-invocation opt-in only -- read fresh every time this
    module is imported (every script/process start), never cached or
    persisted, so enabling a restricted model always requires a conscious
    choice made for that run, not a setting that quietly stays on.
    Comma-separated, e.g. VITALS_ENABLE_RESTRICTED_MODELS=hunyuan."""
    return {name for name in os.environ.get("VITALS_ENABLE_RESTRICTED_MODELS", "").split(",") if name}


REAL_MODEL_BACKENDS = {name for name, meta in MODEL_REGISTRY.items()
                        if not meta["restricted"] or name in _enabled_restricted_models()}
