"""
Train and Evaluate a Tool-Using Agent in PyTorch scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Train and evaluate a tool-using agent.

Story: a teacher writes tool trajectories, a checker filters them, and a small transformer is
fine-tuned on what is left with the tool outputs masked out of the loss. The model then runs as the
agent inside your loop, with sampling, and is scored the way an agent benchmark should be scored:
repeated trials, pass@k next to pass^k, a standard error on every number, a paired comparison of
checkpoints, a split by contamination, and a rerun in a changed environment.

Two agents are trained on the same questions. The "memoriser" saw every trajectory in the same
warehouse. The "tool user" saw a different random warehouse for every trajectory.
"""
import copy
import torch


def pick(pool, per_kind):
    """The first per_kind tasks of each kind."""
    return [t for kind in ("calc", "add", "total") for t in [x for x in pool if x["kind"] == kind][:per_kind]]


def to_tensors(trajectories):
    encoded = [encode_trajectory(t["text"]) for t in trajectories]
    return torch.tensor([e[0] for e in encoded]), torch.tensor([e[1] for e in encoded])


def train_agent(ids, mask, save_at):
    """Supervised fine-tuning from scratch; returns a copy of the model at each step in save_at."""
    torch.manual_seed(0)
    model = ToolLM(len(VOCAB), block_size=40, d_model=48, n_head=4, n_layer=1)
    generator = torch.Generator().manual_seed(0)
    saved, losses, done = {}, [], 0
    for step in save_at:
        losses += train_sft(model, ids, mask, step - done, batch_size=16, lr=3e-3, generator=generator)
        saved[step] = copy.deepcopy(model)
        done = step
    return saved, losses


def evaluate(model, tasks, store, n_trials, max_calls=4):
    """Run the model as the agent at temperature 1.0; one trial is one pass over the benchmark."""
    policy = model_policy(model, temperature=1.0, generator=torch.Generator().manual_seed(1))
    return run_trials(policy, tasks, store, n_trials, max_calls)


def line(name, outcomes):
    s = score_with_se(outcomes)
    return f"{name:<22} score {s['score']:.3f} +- {s['se']:.3f}   one run moves by {s['seed_std']:.3f} between seeds"


def main():
    torch.set_num_threads(1)  # the model is tiny: one thread is faster than many

    # ---- 1. Trajectories from a teacher that is wrong 10% of the time, filtered by the checker ----
    train_tasks = make_tasks(300, seed=0)
    same_store = [STORE] * len(train_tasks)
    random_stores = [random_store(i) for i in range(len(train_tasks))]
    fixed, dropped = collect_trajectories(train_tasks, same_store, make_teacher(0.1, seed=0))
    varied, _ = collect_trajectories(train_tasks, random_stores, make_teacher(0.1, seed=0))
    print(f"teacher trajectories: {len(train_tasks)} tasks, {len(fixed)} kept, {dropped} dropped by the checker")
    print("example:", varied[2]["text"])

    # ---- 2. Supervised fine-tuning with tool outputs masked out of the loss ----
    ids, mask = to_tensors(varied)
    tool_user, losses = train_agent(ids, mask, save_at=(250, 600, 700))
    memoriser, _ = train_agent(*to_tensors(fixed), save_at=(700,))
    result_mask = torch.zeros_like(mask)
    result_mask[:, 1:] = (ids[:, :-1] == STOI["<result>"]).long()
    with torch.no_grad():
        logits = tool_user[700](ids)
        own, tool = sft_loss(logits, ids, mask).item(), sft_loss(logits, ids, result_mask).item()
    print(f"\nloss on the model's own tokens: {sum(losses[:20]) / 20:.2f} at the start, {own:.3f} after 700 steps")
    print(f"loss on tool outputs, which were never a training target: {tool:.2f}")

    # ---- 3. A benchmark: 30 tasks that are also in the training data, 30 that are not ----
    train_questions = [t["question"] for t in varied]
    kept_tasks = [t for t in train_tasks if t["question"] in set(train_questions)]
    fresh = [t for t in make_tasks(400, seed=1) if t["question"] not in set(train_questions)]
    bench = pick(kept_tasks, 10) + pick(fresh, 10)
    runs = {step: evaluate(model, bench, STORE, n_trials=8) for step, model in tool_user.items()}
    final = runs[700]
    print(f"\nbenchmark: {len(bench)} tasks, 8 trials each, temperature 1.0")
    print("tool user, pass@1 of each trial:", score_with_se(final)["seed_scores"])
    for step in runs:
        print(line(f"tool user, step {step}", runs[step]))

    # ---- 4. pass@k (at least one of k trials) against pass^k (all k trials) ----
    for name, outcomes in (("step 250", runs[250]), ("step 700", final)):
        print(f"\n{name}:  k   pass@k   pass^k")
        for k in (1, 2, 4, 8):
            m = pass_k(outcomes, k)
            print(f"           {k}   {m['pass_at_k']:.3f}    {m['pass_hat_k']:.3f}")

    # ---- 5. Is a gap between two checkpoints real? ----
    print()
    for old, new in ((250, 700), (600, 700)):
        c = compare_checkpoints(runs[old], runs[new])
        print(f"step {old} -> {new}: gap {c['gap']:+.3f}, paired SE {c['se_paired']:.3f} "
              f"(unpaired {c['se_unpaired']:.3f}): {c['verdict']}")

    # ---- 6. Contamination: tasks seen in training against held-out tasks ----
    memo_run = evaluate(memoriser[700], bench, STORE, n_trials=8)
    print()
    for name, outcomes in (("tool user", final), ("memoriser", memo_run)):
        r = contamination_report(bench, outcomes, train_questions)
        print(f"{name}: seen {r['seen']['score']:.3f} (n={r['seen']['n']}), held out {r['held_out']['score']:.3f} "
              f"(n={r['held_out']['n']}), gap {r['gap']:+.3f} +- {r['se_gap']:.3f}")

    # ---- 7. A perturbed benchmark: same questions, the warehouse after a delivery ----
    print()
    for name, model, before in (("tool user", tool_user[700], final), ("memoriser", memoriser[700], memo_run)):
        after = evaluate(model, bench, RESTOCKED, n_trials=4)
        print(f"{name}: {score_with_se(before)['score']:.3f} in the training warehouse, "
              f"{score_with_se(after)['score']:.3f} after the stocks change")

    # ---- 8. The harness is part of the score ----
    tight = evaluate(tool_user[700], bench, STORE, n_trials=4, max_calls=2)
    print(f"\ntool user with a budget of 2 tool calls instead of 4: {score_with_se(tight)['score']:.3f}")

    episode = run_episode(model_policy(tool_user[700], 1.0, torch.Generator().manual_seed(5)), bench[-1], STORE)
    print("\none episode:", episode["transcript"])
    print("status:", episode["status"], "| correct:", episode["correct"])


if __name__ == "__main__":
    main()

