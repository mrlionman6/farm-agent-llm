"""
make_farm_data.py
=================
Build the training / test data for the farm agent. No hand-written data.

For each sample:
  1. random task (e.g. "clear" + "plant") and random start state
  2. English command (random wording)
  3. oracle plan from farm_tasks.oracle_plan
  4. KEEP only if the plan passes validation AND really achieves the goal
     when executed in the simulator

Splits (all disjoint by seed):
  train.jsonl        {"text": prompt + plan}            <- for train.py
  test_seen.jsonl    new states, phrasings seen in training
  test_newphr.jsonl  new states AND a phrasing never seen in training
Test rows also store the start-state spec so evaluate.py can replay them.

Run:
    python make_farm_data.py
"""

import json
import os
import random

import farm_env as F
import farm_tasks as T

OUT_DIR = "data"
N_TRAIN, N_TEST_SEEN, N_TEST_NEW = 4000, 300, 300
SEED = 0


def build_one(rng, held_out):
    for _ in range(200):                      # retry until a verifiable sample appears
        ops = T.sample_task(rng)
        spec = T.sample_state(rng, [o["op"] for o in ops])
        env0 = T.make_env(spec)
        plan = T.oracle_plan(env0, ops)

        # verify on a FRESH world, through the same control layer the LLM will use
        env = T.make_env(spec)
        res = F.run_plan(env, plan)
        ok, _ = T.check_goals(spec, ops, env)
        if not (res["valid"] and res["completed"] and ok):
            continue
        command = T.render_command(ops, rng, held_out=held_out)
        prompt = T.make_prompt(command, env0.observe_text())
        return {"prompt": prompt, "plan": plan, "ops": ops, "spec": spec, "command": command}
    raise RuntimeError("could not build a sample")


def write(path, rows, as_text):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            out = {"text": r["prompt"] + T.plan_to_text(r["plan"])} if as_text else r
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    seen_specs = set()

    def collect(n, held_out):
        rows = []
        while len(rows) < n:
            r = build_one(rng, held_out)
            key = (r["spec"]["seed"], json.dumps(r["spec"], sort_keys=True), r["command"])
            if r["spec"]["seed"] in seen_specs or key in seen_specs:
                continue
            seen_specs.add(r["spec"]["seed"])      # one sample per world seed => no leakage
            rows.append(r)
        return rows

    train = collect(N_TRAIN, held_out=False)
    test_seen = collect(N_TEST_SEEN, held_out=False)
    test_new = collect(N_TEST_NEW, held_out=True)

    write(os.path.join(OUT_DIR, "farm_train.jsonl"), train, as_text=True)
    write(os.path.join(OUT_DIR, "farm_test_seen.jsonl"), test_seen, as_text=False)
    write(os.path.join(OUT_DIR, "farm_test_newphr.jsonl"), test_new, as_text=False)

    lens = [len(T.plan_to_text(r["plan"])) for r in train]
    print(f"train          : {len(train)}  -> {OUT_DIR}/farm_train.jsonl")
    print(f"test_seen      : {len(test_seen)}  -> {OUT_DIR}/farm_test_seen.jsonl")
    print(f"test_newphr    : {len(test_new)}  -> {OUT_DIR}/farm_test_newphr.jsonl")
    print(f"plan length (chars)  avg {sum(lens)/len(lens):.0f}  max {max(lens)}")


if __name__ == "__main__":
    main()
