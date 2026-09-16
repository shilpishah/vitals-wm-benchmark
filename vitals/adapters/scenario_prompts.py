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
    # Reuses occlusion_corridor_distractor's own scene file AS-IS (that
    # manifest's own comment) -- byte-identical prompt, since a real video
    # model only ever sees rendered PIXELS, never a manifest's own
    # target_property (P2 vs P3), and the input-normalization protocol
    # describes what's actually IN the scene, not which detector is
    # scoring it.
    "occlusion_corridor_interpenetration": (
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
    "collision": (
        "A red ball rolls across a flat gray floor and strikes a "
        "stationary blue ball, transferring momentum on impact, under "
        "realistic physics."
    ),
    "billiards": (
        "A red ball rolls across a flat gray floor and strikes a "
        "stationary blue ball, transferring momentum on impact. The blue "
        "ball then rolls forward and strikes a stationary green ball "
        "further along, transferring momentum again, under realistic "
        "physics."
    ),
    # 2026-09-11, scenario scaling. Same plain-description discipline: the
    # wall is long, the ball goes fully out of view behind it and comes
    # back out the far side -- stated because that IS what the scene does,
    # not to coach the model into re-emergence.
    "occlusion_reemergence": (
        "A red ball rolls in a straight line across a flat gray floor, "
        "passes completely out of view behind a long solid gray wall, and "
        "then rolls back into view on the far side of the wall, continuing "
        "in the same direction under realistic physics."
    ),
    # Render-domain intervention (M10, 2026-09-16): the same sentence with
    # only the colours/material words changed to what the frames show.
    "occlusion_reemergence_domA": (
        "A green ball rolls in a straight line across a flat wooden floor, "
        "passes completely out of view behind a long solid dark-blue wall, and "
        "then rolls back into view on the far side of the wall, continuing "
        "in the same direction under realistic physics."
    ),
    "occlusion_reemergence_domB": (
        "A red ball rolls in a straight line across a flat gray floor, "
        "passes completely out of view behind a long solid gray wall, and "
        "then rolls back into view on the far side of the wall, continuing "
        "in the same direction under realistic physics."
    ),
    "block_stack": (
        "A red ball rolls across a flat gray floor and strikes a blue cube "
        "with a tall yellow block standing upright on top of it. The impact "
        "pushes the cube and the tall block topples off, under realistic "
        "physics."
    ),
    "soft_ramp": (
        "A soft red rubber ball rolls down a wooden ramp onto a flat gray "
        "floor, squashing slightly as it rolls, and continues rolling across "
        "the floor while slowing down, keeping its size and volume, under "
        "realistic physics."
    ),
    "soft_drop": (
        "A soft red rubber ball falls onto a flat gray floor while moving "
        "sideways, squashes on impact, bounces, and rolls on, keeping its "
        "size and volume, under realistic physics."
    ),
}
