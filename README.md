# Train and Evaluate a Tool-Using Agent in PyTorch

Train a small language model to use tools, run it as an agent, and measure how good it really is. Write a calculator and a lookup tool, a text format for tool calls, and the loop that parses a call, runs the tool and feeds the result back, with a checker for the final answer. Collect trajectories from a teacher, keep the ones the checker accepts, and fine-tune a small transformer on them with the tool outputs masked out of the loss. Then run the model as the agent with sampling and evaluate it over repeated trials: pass@k next to pass^k, a standard error on every score, a paired comparison of two checkpoints, a split between tasks seen in training and held-out tasks, and a rerun after the environment changes.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** run_tool
- [x] **2.** parse_action
- [x] **3.** run_episode
- [x] **4.** collect_trajectories
- [x] **5.** encode_trajectory
- [x] **6.** ToolLM
- [x] **7.** sft_loss
- [x] **8.** train_sft
- [x] **9.** model_policy
- [x] **10.** run_trials
- [x] **11.** pass_k
- [ ] **12.** score_with_se
- [ ] **13.** compare_checkpoints
- [ ] **14.** contamination_report

---

Built on Deep-ML.
