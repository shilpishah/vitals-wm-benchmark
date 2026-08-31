"""One frozen, physically-descriptive text prompt per scenario, SHARED
across every text-conditioned video model (AGENT.md M6) -- extracted from
`cosmos.py` (2026-08, when a second model, Wan2.1, needed the identical
prompts) rather than letting each model file keep its own copy. Using the
literal SAME prompt text for the SAME scenario across every model is not
optional: the input-normalization protocol's own point is "compared only
within matched settings," and two models scored against subtly different
wording would not be a fair comparison, no matter how small the
difference looked.

Input-normalization protocol discipline (decide once, do not adjust per
episode or per model): plain descriptions of what's actually in each
scene (scenes/*.xml's own geometry/materials), not adjectives aimed at
steering any one model toward a "nicer" result.
"""
SCENARIO_PROMPTS = {
    "occlusion_corridor": (
        "A red ball rolls in a straight line across a flat gray floor, "
        "briefly passing behind a solid gray wall before continuing to roll "
        "on the other side, under realistic physics."
    ),
    "occlusion_corridor_distractor": (
        "A red ball rolls in a straight line across a flat gray floor, "
        "briefly passing behind a solid gray wall before continuing to roll "
        "on the other side, under realistic physics. A separate object "
        "sits elsewhere in the scene without interacting with the ball."
    ),
    "occlusion_corridor_moving": (
        "A red ball rolls in a straight line across a flat gray floor. A "
        "separate gray object slides across the floor independently, "
        "sometimes passing between the camera and the ball. Both move "
        "under realistic physics."
    ),
    "ramp_descent": (
        "A red ball slides and rolls down a gray inclined ramp onto a flat "
        "gray floor, continuing to roll under realistic physics and gravity."
    ),
    "ramp_descent_high_friction": (
        "A red ball slides and rolls down a gray inclined ramp onto a flat "
        "gray floor, quickly slowing down and coming to a stop under "
        "realistic physics and gravity."
    ),
    "projectile": (
        "A red ball is thrown through the air and falls under gravity, "
        "bouncing on a flat gray floor several times, losing height with "
        "each bounce, under realistic physics."
    ),
}
