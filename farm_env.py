"""
farm_env.py
===========
"Mini Farm": a tiny Stardew-Valley-style environment for testing an LLM that
turns English commands into PLANS made of a fixed set of skills.

Layers (the LLM only ever touches the first one):
  1. validate_plan()  - control layer: allowlist of skills, argument checks, step limit
  2. run_plan()       - executes a validated plan, step by step, stops at first failure
  3. FarmEnv          - the world: time, energy, money, shop hours, debris, crops

The LLM never presses keys. It writes a plan like:
  {"plan": [
      {"skill": "go_to",      "args": {"location": "farm"}},
      {"skill": "clear_area", "args": {"kind": "all"}},
      {"skill": "wait_until", "args": {"time": "09:00"}},
      {"skill": "go_to",      "args": {"location": "shop"}},
      {"skill": "buy",        "args": {"crop": "parsnip", "qty": 5}},
      {"skill": "go_to",      "args": {"location": "farm"}},
      {"skill": "plant",      "args": {"crop": "parsnip", "qty": 5}},
      {"skill": "water",      "args": {}}
  ]}

All numbers (prices, times, distances) are made up for this simulation.
"""

import json
import random
import re

# --------------------------------------------------------------------------
# World constants
# --------------------------------------------------------------------------
SKILLS = ("wait_until", "go_to", "clear_area", "buy", "plant", "water", "sleep", "harvest")
LOCATIONS = ("home", "farm", "shop")
_TRAVEL_MIN = {
    frozenset(("home", "farm")): 10,
    frozenset(("farm", "shop")): 20,
    frozenset(("home", "shop")): 30,
}

DEBRIS_KINDS = ("weed", "rock", "branch")
CLEAR_MIN = {"weed": 5, "rock": 15, "branch": 10}      # minutes per tile
CLEAR_ENERGY = {"weed": 1, "rock": 3, "branch": 2}     # energy per tile

CROPS = {
    "parsnip": {"price": 20, "days": 4},
    "potato": {"price": 50, "days": 6},
    "cauliflower": {"price": 80, "days": 7},
}

GRID_W, GRID_H = 6, 4
START_MIN = 6 * 60          # 06:00
END_MIN = 26 * 60           # 02:00 next morning: day is over
SHOP_OPEN, SHOP_CLOSE = 9 * 60, 17 * 60
SHOP_CLOSED_WEEKDAY = 3     # shop is closed on days where day % 7 == 3
MAX_ENERGY = 100
MAX_STEPS = 20              # control layer: longest plan we will accept


# --------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------
def fmt_time(minutes):
    return f"{(minutes // 60) % 24:02d}:{minutes % 60:02d}"


def parse_time(s):
    """'HH:MM' -> minutes since midnight. Valid range 06:00-25:59, else None."""
    if not isinstance(s, str):
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})", s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if mi > 59 or not (6 <= h <= 25):
        return None
    return h * 60 + mi


def _r(ok, reason=None, **detail):
    return {"ok": ok, "reason": reason, "detail": detail}


# --------------------------------------------------------------------------
# The world
# --------------------------------------------------------------------------
class FarmEnv:
    def __init__(self, seed=0, money=500):
        rng = random.Random(seed)
        self.day = 1
        self.t = START_MIN
        self.loc = "home"
        self.money = money
        self.energy = MAX_ENERGY
        self.inv = {}
        self.tiles = []
        for _ in range(GRID_W * GRID_H):
            debris = rng.choice(DEBRIS_KINDS) if rng.random() < 0.5 else None
            self.tiles.append({"debris": debris, "crop": None, "growth": 0, "watered": False})

    # ---- state views ------------------------------------------------------
    def shop_is_open(self):
        return self.day % 7 != SHOP_CLOSED_WEEKDAY and SHOP_OPEN <= self.t < SHOP_CLOSE

    def summary(self):
        debris = {k: 0 for k in DEBRIS_KINDS}
        crops = {}
        for tile in self.tiles:
            if tile["debris"]:
                debris[tile["debris"]] += 1
            if tile["crop"]:
                c = crops.setdefault(tile["crop"], {"planted": 0, "watered": 0, "ready": 0})
                c["planted"] += 1
                c["watered"] += int(tile["watered"])
                c["ready"] += int(tile["growth"] >= CROPS[tile["crop"]]["days"])
        return {
            "day": self.day, "time": fmt_time(self.t), "location": self.loc,
            "money": self.money, "energy": self.energy,
            "debris": debris, "debris_total": sum(debris.values()),
            "crops": crops, "inventory": dict(self.inv),
            "shop_open_now": self.shop_is_open(),
        }

    def observe_text(self):
        """Compact text of the current state. This is what the LLM will see."""
        s = self.summary()
        debris = ", ".join(f"{k} {v}" for k, v in s["debris"].items() if v) or "none"
        crops = ", ".join(
            f"{k} {v['planted']} (watered {v['watered']}, ready {v['ready']})"
            for k, v in s["crops"].items()
        ) or "none"
        inv = ", ".join(f"{k} {v}" for k, v in s["inventory"].items()) or "empty"
        closed_today = self.day % 7 == SHOP_CLOSED_WEEKDAY
        shop = "closed today" if closed_today else f"open {fmt_time(SHOP_OPEN)}-{fmt_time(SHOP_CLOSE)}"
        return (
            f"Day {s['day']}, {s['time']}. Location: {s['location']}. "
            f"Money: {s['money']}g. Energy: {s['energy']}/{MAX_ENERGY}.\n"
            f"Farm ({GRID_W}x{GRID_H} tiles): debris: {debris}. Crops: {crops}.\n"
            f"Inventory: {inv}.\n"
            f"Shop: {shop}."
        )

    # ---- dispatch ---------------------------------------------------------
    def step(self, skill, args):
        if skill not in SKILLS:
            return _r(False, f"unknown skill '{skill}'")
        try:
            return getattr(self, f"do_{skill}")(**args)
        except TypeError as e:
            return _r(False, f"bad arguments: {e}")

    # ---- skills -----------------------------------------------------------
    def do_wait_until(self, time):
        target = parse_time(time)
        if target is None:
            return _r(False, "invalid time")
        if target <= self.t:
            return _r(True, None, note="already past, nothing to wait for", now=fmt_time(self.t))
        self.t = target
        return _r(True, None, now=fmt_time(self.t))

    def do_go_to(self, location):
        if location not in LOCATIONS:
            return _r(False, f"unknown location '{location}'")
        if location == self.loc:
            return _r(True, None, note="already there")
        minutes = _TRAVEL_MIN[frozenset((self.loc, location))]
        if self.t + minutes > END_MIN:
            return _r(False, "too late in the day to travel")
        self.t += minutes
        self.loc = location
        return _r(True, None, now=fmt_time(self.t))

    def do_clear_area(self, kind):
        if self.loc != "farm":
            return _r(False, "must be at the farm")
        if kind != "all" and kind not in DEBRIS_KINDS:
            return _r(False, f"unknown debris kind '{kind}'")
        targets = [
            i for i, tile in enumerate(self.tiles)
            if tile["debris"] and (kind == "all" or tile["debris"] == kind)
        ]
        if not targets:
            return _r(True, None, cleared=0, note="nothing to clear")
        cleared, reason = 0, None
        for i in targets:
            d = self.tiles[i]["debris"]
            if self.energy < CLEAR_ENERGY[d]:
                reason = "out of energy"
                break
            if self.t + CLEAR_MIN[d] > END_MIN:
                reason = "out of time today"
                break
            self.energy -= CLEAR_ENERGY[d]
            self.t += CLEAR_MIN[d]
            self.tiles[i]["debris"] = None
            cleared += 1
        if cleared == len(targets):
            return _r(True, None, cleared=cleared, now=fmt_time(self.t))
        return _r(False, f"{reason} after clearing {cleared} of {len(targets)} tiles",
                  cleared=cleared, now=fmt_time(self.t))

    def do_buy(self, crop, qty):
        if self.loc != "shop":
            return _r(False, "must be at the shop")
        if crop not in CROPS:
            return _r(False, f"shop does not sell '{crop}'")
        if not self.shop_is_open():
            if self.day % 7 == SHOP_CLOSED_WEEKDAY:
                return _r(False, "shop is closed today")
            return _r(False, f"shop is closed now (open {fmt_time(SHOP_OPEN)}-{fmt_time(SHOP_CLOSE)})")
        cost = CROPS[crop]["price"] * qty
        if cost > self.money:
            return _r(False, f"not enough money: need {cost}g, have {self.money}g")
        if self.t + 5 > END_MIN:
            return _r(False, "out of time today")
        self.money -= cost
        self.t += 5
        key = f"{crop}_seed"
        self.inv[key] = self.inv.get(key, 0) + qty
        return _r(True, None, bought=qty, spent=cost, now=fmt_time(self.t))

    def do_plant(self, crop, qty):
        if self.loc != "farm":
            return _r(False, "must be at the farm")
        if crop not in CROPS:
            return _r(False, f"unknown crop '{crop}'")
        seeds = self.inv.get(f"{crop}_seed", 0)
        free = [i for i, t in enumerate(self.tiles) if t["debris"] is None and t["crop"] is None]
        want = seeds if qty == "all" else qty
        target = min(want, seeds, len(free))
        planted = 0
        for i in free[:target]:
            if self.energy < 1 or self.t + 2 > END_MIN:
                break
            self.energy -= 1
            self.t += 2
            self.tiles[i].update(crop=crop, growth=0, watered=False)
            planted += 1
        self.inv[f"{crop}_seed"] = seeds - planted
        if qty == "all":
            if planted > 0:
                return _r(True, None, planted=planted, now=fmt_time(self.t))
            why = "no seeds" if seeds == 0 else "no free tiles" if not free else "out of energy or time"
            return _r(False, why, planted=0)
        if planted == qty:
            return _r(True, None, planted=planted, now=fmt_time(self.t))
        return _r(False, f"planted {planted} of {qty} (seeds: {seeds}, free tiles: {len(free)})",
                  planted=planted)

    def do_water(self):
        if self.loc != "farm":
            return _r(False, "must be at the farm")
        planted = [t for t in self.tiles if t["crop"]]
        if not planted:
            return _r(False, "no crops planted")
        todo = [t for t in planted if not t["watered"]]
        watered = 0
        for tile in todo:
            if self.energy < 1 or self.t + 1 > END_MIN:
                break
            self.energy -= 1
            self.t += 1
            tile["watered"] = True
            watered += 1
        if watered == len(todo):
            return _r(True, None, watered=watered, now=fmt_time(self.t))
        return _r(False, f"out of energy or time after watering {watered} of {len(todo)}",
                  watered=watered)

    def do_sleep(self):
        if self.loc != "home":
            return _r(False, "must be at home")
        for tile in self.tiles:
            if tile["crop"] and tile["watered"]:
                tile["growth"] += 1
            tile["watered"] = False
        self.day += 1
        self.t = START_MIN
        self.energy = MAX_ENERGY
        return _r(True, None, day=self.day)

    def do_harvest(self):
        if self.loc != "farm":
            return _r(False, "must be at the farm")
        ready = [t for t in self.tiles if t["crop"] and t["growth"] >= CROPS[t["crop"]]["days"]]
        if not ready:
            return _r(False, "nothing ready to harvest")
        got = {}
        for tile in ready:
            if self.t + 2 > END_MIN:
                break
            self.t += 2
            got[tile["crop"]] = got.get(tile["crop"], 0) + 1
            self.inv[tile["crop"]] = self.inv.get(tile["crop"], 0) + 1
            tile.update(crop=None, growth=0, watered=False)
        return _r(True, None, harvested=got, now=fmt_time(self.t))


# --------------------------------------------------------------------------
# Control layer: validate a plan BEFORE anything runs
# --------------------------------------------------------------------------
def _qty_int(v):
    return isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 99


ARG_SPECS = {
    "wait_until": {"time": lambda v: parse_time(v) is not None},
    "go_to": {"location": lambda v: v in LOCATIONS},
    "clear_area": {"kind": lambda v: v in ("all",) + DEBRIS_KINDS},
    "buy": {"crop": lambda v: v in CROPS, "qty": _qty_int},
    "plant": {"crop": lambda v: v in CROPS, "qty": lambda v: v == "all" or _qty_int(v)},
    "water": {},
    "sleep": {},
    "harvest": {},
}


def validate_plan(plan_obj):
    """Return a list of error strings. Empty list = plan is structurally valid."""
    if not isinstance(plan_obj, dict) or "plan" not in plan_obj:
        return ["top level must be an object with a 'plan' list"]
    steps = plan_obj["plan"]
    if not isinstance(steps, list):
        return ["'plan' must be a list"]
    if not 1 <= len(steps) <= MAX_STEPS:
        return [f"plan must have 1-{MAX_STEPS} steps, got {len(steps)}"]
    errors = []
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict) or set(step) != {"skill", "args"}:
            errors.append(f"step {i}: must have exactly 'skill' and 'args'")
            continue
        skill, args = step["skill"], step["args"]
        if skill not in SKILLS:
            errors.append(f"step {i}: skill '{skill}' is not allowed")
            continue
        if not isinstance(args, dict):
            errors.append(f"step {i}: 'args' must be an object")
            continue
        spec = ARG_SPECS[skill]
        if set(args) != set(spec):
            errors.append(f"step {i}: {skill} needs args {sorted(spec)}, got {sorted(args)}")
            continue
        for name, ok in spec.items():
            if not ok(args[name]):
                errors.append(f"step {i}: invalid value for {skill}.{name}: {args[name]!r}")
    return errors


def parse_plan(text):
    """Pull a JSON object out of model output (tolerates extra text / code fences).
    Returns (plan_obj, None) or (None, error_message)."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None, "no JSON object found"
    try:
        return json.loads(text[start:end + 1]), None
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------
def run_plan(env, plan_obj, stop_on_fail=True):
    """Validate, then execute step by step.
    Returns {"valid", "errors", "results", "completed"}.
    An invalid plan never touches the environment."""
    errors = validate_plan(plan_obj)
    if errors:
        return {"valid": False, "errors": errors, "results": [], "completed": False}
    results = []
    for step in plan_obj["plan"]:
        res = env.step(step["skill"], step["args"])
        results.append({"skill": step["skill"], "args": step["args"], **res})
        if not res["ok"] and stop_on_fail:
            return {"valid": True, "errors": [], "results": results, "completed": False}
    return {"valid": True, "errors": [], "results": results, "completed": all(r["ok"] for r in results)}


if __name__ == "__main__":
    env = FarmEnv(seed=0)
    print(env.observe_text())
