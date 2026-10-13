"""
farm_tasks.py
=============
Everything about TASKS for the Mini Farm (shared by make_farm_data.py and,
later, evaluate.py):

  - make_env(spec)      rebuild a start state from a small JSON-able dict
  - sample_task / sample_state / render_command   random task + state + English wording
  - oracle_plan(...)    a rule-based planner that writes the "correct" plan
  - check_goals(...)    did the executed plan actually achieve the goal?
  - make_prompt(...)    the exact text the LLM sees (training and inference)

The oracle is only used to create training labels (and as a baseline).
A sample is kept ONLY if executing its oracle plan in the simulator really
achieves the goal, so labels are verified, not just written.
"""

import copy
import json
import random

import farm_env as F

# --------------------------------------------------------------------------
# Wording. The LAST template of each list is held out of training, so the
# test set can ask: "does the model cope with a phrasing it never saw?"
# --------------------------------------------------------------------------
PLURAL = {"weed": "weeds", "rock": "rocks", "branch": "branches",
          "parsnip": "parsnips", "potato": "potatoes", "cauliflower": "cauliflowers"}

PHRASES = {
    "clear_all": [
        "Clear the farm.",
        "Clean up my farm.",
        "Get rid of all the debris on the farm.",
        "Please clear everything off the farm.",            # held out
    ],
    "clear_kind": [
        "Clear the {kinds} from the farm.",
        "Remove all {kinds}.",
        "Get rid of the {kinds} on the farm.",
        "Take care of the {kinds} on my farm.",             # held out
    ],
    "plant": [
        "Plant {n} {crops}.",
        "Buy {n} {crop} seeds and plant them.",
        "I want {n} {crops} growing on the farm.",
        "Get some {crops} in the ground, {n} of them.",     # held out
    ],
    "water": [
        "Water the crops.",
        "Water everything.",
        "Give the plants some water.",
        "Make sure my crops get watered.",                  # held out
    ],
    "harvest": [
        "Harvest the crops.",
        "Collect whatever is ready.",
        "Pick the ripe crops.",
        "Bring in the harvest.",                            # held out
    ],
    "sleep": [
        "Go to sleep.",
        "End the day.",
        "Time for bed.",
        "Call it a night.",                                 # held out
    ],
}

# Each task is a list of ops, done in this order. (name, weight)
TASK_KINDS = [
    (("clear_all",), 10),
    (("clear_kind",), 8),
    (("plant",), 14),
    (("water",), 6),
    (("harvest",), 6),
    (("sleep",), 3),
    (("clear_all", "plant"), 12),
    (("clear_kind", "plant"), 5),
    (("plant", "water"), 8),
    (("clear_all", "plant", "water"), 8),
    (("water", "sleep"), 3),
    (("harvest", "sleep"), 3),
]


# --------------------------------------------------------------------------
# Start states
# --------------------------------------------------------------------------
def make_env(spec):
    """Rebuild a FarmEnv from a spec dict (so test rows can be replayed exactly)."""
    env = F.FarmEnv(seed=spec["seed"], money=spec["money"])
    env.day, env.t, env.loc, env.energy = spec["day"], spec["t"], spec["loc"], spec["energy"]
    env.inv = dict(spec.get("inv", {}))
    c = spec.get("crops")
    if c:
        for tile in env.tiles[: c["n"]]:
            tile.update(debris=None, crop=c["crop"], growth=c["growth"], watered=c["watered"])
    return env


def sample_state(rng, ops):
    spec = {
        "seed": rng.randrange(10**6),
        "money": rng.randrange(60, 601, 10),
        "day": rng.randint(1, 10),
        "t": rng.choice(range(360, 931, 30)),
        "loc": rng.choice(F.LOCATIONS),
        "energy": rng.choice([100, 100, 100, 80, 60, 40]),
        "inv": {},
    }
    if rng.random() < 0.25:                         # sometimes already own some seeds
        spec["inv"] = {f"{rng.choice(list(F.CROPS))}_seed": rng.randint(1, 6)}
    needs_crops = "water" in ops or "harvest" in ops
    if needs_crops or rng.random() < 0.3:
        crop = rng.choice(list(F.CROPS))
        days = F.CROPS[crop]["days"]
        if "harvest" in ops:
            growth = days                           # ready
        elif "water" in ops and "plant" not in ops:
            growth = rng.randint(0, days - 1)
        else:
            growth = rng.randint(0, days)
        spec["crops"] = {"crop": crop, "n": rng.randint(1, 6), "growth": growth,
                         "watered": False if "water" in ops else rng.random() < 0.5}
    return spec


def sample_task(rng):
    kinds, weights = zip(*TASK_KINDS)
    ops_names = rng.choices(kinds, weights)[0]
    ops = []
    for name in ops_names:
        if name == "clear_kind":
            ops.append({"op": "clear", "kind": rng.choice(F.DEBRIS_KINDS)})
        elif name == "clear_all":
            ops.append({"op": "clear", "kind": "all"})
        elif name == "plant":
            ops.append({"op": "plant", "crop": rng.choice(list(F.CROPS)), "n": rng.randint(2, 8)})
        else:
            ops.append({"op": name})
    return ops


def _phrase_key(op):
    if op["op"] == "clear":
        return "clear_all" if op["kind"] == "all" else "clear_kind"
    return op["op"]


def render_command(ops, rng, held_out=False):
    """English command for a task. held_out=True forces the held-out template for at
    least one op; held_out=False only uses training templates."""
    parts, used_held = [], False
    for i, op in enumerate(ops):
        key = _phrase_key(op)
        templates = PHRASES[key]
        if held_out and (i == len(ops) - 1 or rng.random() < 0.3) and not used_held:
            tpl, used_held = templates[-1], True
        else:
            tpl = rng.choice(templates[:-1])
        text = tpl.format(
            kinds=PLURAL.get(op.get("kind"), ""), n=op.get("n", ""),
            crops=PLURAL.get(op.get("crop"), ""), crop=op.get("crop", ""),
        )
        parts.append(text if i == 0 else text[0].lower() + text[1:])
    return " Then ".join(parts) if len(parts) > 1 else parts[0]


# --------------------------------------------------------------------------
# Oracle planner: builds the plan by *simulating* each step on a copy of the
# world, so waits and trips are always computed from the real state.
# --------------------------------------------------------------------------
class _Planner:
    def __init__(self, env):
        self.env = copy.deepcopy(env)
        self.plan = []

    def emit(self, skill, **args):
        self.plan.append({"skill": skill, "args": args})
        self.env.step(skill, args)

    def at(self, loc):
        if self.env.loc != loc:
            self.emit("go_to", location=loc)

    def shop_for(self, crop, qty):
        e = self.env
        if e.day % 7 == F.SHOP_CLOSED_WEEKDAY:      # closed all day: sleep, shop tomorrow
            self.at("home")
            self.emit("sleep")
        self.at("shop")
        if self.env.t < F.SHOP_OPEN:
            self.emit("wait_until", time=F.fmt_time(F.SHOP_OPEN))
        self.emit("buy", crop=crop, qty=qty)

    def do(self, op):
        k = op["op"]
        if k == "clear":
            self.at("farm")
            self.emit("clear_area", kind=op["kind"])
        elif k == "plant":
            have = self.env.inv.get(f"{op['crop']}_seed", 0)
            if have < op["n"]:
                self.shop_for(op["crop"], op["n"] - have)
            self.at("farm")
            self.emit("plant", crop=op["crop"], qty=op["n"])
        elif k == "water":
            self.at("farm")
            self.emit("water")
        elif k == "harvest":
            self.at("farm")
            self.emit("harvest")
        elif k == "sleep":
            self.at("home")
            self.emit("sleep")


def oracle_plan(env, ops):
    p = _Planner(env)
    for op in ops:
        p.do(op)
    return {"plan": p.plan}


# --------------------------------------------------------------------------
# Goal checking (on the REAL world after the plan ran)
# --------------------------------------------------------------------------
def check_goals(spec, ops, env_after):
    """Return (ok, [failed goal descriptions]) comparing start spec -> end state."""
    before = make_env(spec).summary()
    after = env_after.summary()
    failed = []
    for op in ops:
        k = op["op"]
        if k == "clear":
            left = after["debris_total"] if op["kind"] == "all" else after["debris"][op["kind"]]
            if left:
                failed.append(f"clear {op['kind']}: {left} left")
        elif k == "plant":
            c = op["crop"]
            was = before["crops"].get(c, {}).get("planted", 0)
            now = after["crops"].get(c, {}).get("planted", 0)
            if now < was + op["n"]:
                failed.append(f"plant {c}: wanted +{op['n']}, got +{now - was}")
        elif k == "water":
            tot = sum(v["planted"] for v in after["crops"].values())
            wat = sum(v["watered"] for v in after["crops"].values())
            if tot == 0 or wat < tot:
                failed.append(f"water: {wat}/{tot} watered")
        elif k == "harvest":
            if any(v["ready"] for v in after["crops"].values()):
                failed.append("harvest: ready crops remain")
            elif sum(after["inventory"].get(c, 0) for c in F.CROPS) <= \
                    sum(before["inventory"].get(c, 0) for c in F.CROPS):
                failed.append("harvest: nothing collected")
        elif k == "sleep":
            if after["day"] <= before["day"]:
                failed.append("sleep: day did not advance")
    return (not failed), failed


# --------------------------------------------------------------------------
# Prompt / plan text. Compact JSON keeps the sequence short for a small model.
# --------------------------------------------------------------------------
def make_prompt(command, state_text):
    return f"### Command:\n{command}\n### State:\n{state_text}\n### Plan:\n"


def plan_to_text(plan_obj):
    return json.dumps(plan_obj, separators=(",", ":"))
