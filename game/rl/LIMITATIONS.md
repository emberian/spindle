# RIG RL — Known Limitations

## The gradient bottleneck (highest priority)

The MAPPO trainer uses **ES-shaped PPO** — it computes the PPO clipped-surrogate
objective correctly but estimates the gradient via evolutionary strategy perturbations
rather than analytical backpropagation. At 24,598 actor params this means:

- Each "gradient step" requires `es_pop_size` (default 20) full rollout evaluations
  just to estimate the direction, vs 1 forward+backward pass with true backprop
- Sample efficiency is ~20-100x worse than PyTorch PPO at equal param count
- The value function (34k params) also uses ES, which is particularly wasteful for
  a simple 2-layer MLP where backprop is textbook

**What's needed:** Analytical backward pass for at least the critic (chain-ruled
matmul + tanh derivatives — straightforward) and ideally the actor (attention
backward through softmax + residual + layer norm is harder but well-documented).
This is the single highest-leverage improvement for scaling to massive runs.

**Alternatively:** Use the Python FFI bridge + PyTorch/JAX for training (the bridge
exists and works at 8250 t/s) and only deploy the trained attention weights to the
Rust forward pass. This separates "training framework" from "inference runtime" —
standard practice in production ML.

## The keep-away basin

The ES learner (at feasible budgets) reliably finds "elegant keep-away":
- `spatial_control = 1.0` (total possession)
- `pass_chain_depth = 1499` (passing among teammates continuously)
- `gate_pursuit = 0` (never advancing the cast)
- `net_score = 0` (never actually scoring)

The policy-invariant shaping (gate-progress potential) exists but at current budgets
the trivially-reachable pass-chain term dominates. Escaping this basin requires:
1. More compute (longer ES/MAPPO runs with larger populations)
2. Reweighting Phi to favor gate-progress over raw pass-chain
3. Possibly curriculum: start with dense gate reward, anneal toward pure-intrinsic

## Structural observation gaps

The attention policy sees:
- Its own role + Director assignment
- Real gate-plane geometry (not just an index)
- Nearest opponent closing speed
- Score/clock context

It does NOT see:
- Opponent's assignment/intent (partial observability — by design, but makes defense harder to learn)
- Full line state of opponents (only own line)
- Historical trajectory (no recurrence/memory — each decision is Markov on current obs)

## Action space limitations

- **No targeted throw direction.** The "pass-to-teammate-k" head targets the k-th
  nearest teammate's lead point, but there's no "throw to empty space for a run" action.
  Gate-clearing requires throwing to a point the receiver will reach, not where they are.
- **Fixed-std Gaussian for continuous actions.** The std is a hyperparameter, not learned.
  The policy can't express high-confidence precise aims vs exploratory wide aims.
- **No communication channel.** Agents can't signal intent to teammates. Coordination
  must emerge purely from shared weights + positional coincidence.

## Sim model gaps affecting learning

From the critical audit (these affect what the learned policy can discover):

- **Bell spin is decoratively modeled but gameplay-irrelevant.** A tumbling bell is
  exactly as catchable as a true one. The policy can't learn to exploit/avoid spin.
- **The powered winch is free and unlimited.** No energy/stamina budget → no resource
  coordination problem. The policy can winch forever with no tradeoff.
- **`grounded` is permanent.** A rigger that touches the wall is deleted from the
  contest forever. The policy should learn to avoid the skin, but recovery isn't possible.
- **Strip is a global-counter heuristic.** The learned defense can approach to strip
  range, but the actual strip mechanic is a magic-number timer, not a force model.

## Test coverage gaps

- No test asserts that the attention policy LEARNS (only that it runs without panics
  and produces valid actions). The ES learn-proof tests are for the MLP policy.
- No integration test drives `train_mappo` end-to-end from a test harness (the binary
  is tested by running it manually).
- The dashboard has no automated tests (it's a dev tool with sample data; correctness
  is verified by the user watching).

## Scale limitations

- **Single-machine only.** Rayon parallelizes across cores on one machine. A "massive
  arena-style" run across multiple machines needs distributed rollout collection.
- **No checkpointing/resume.** If training crashes, it restarts from scratch (the
  population is in-memory only; not serialized to disk between generations).
- **Fixed team size.** The attention policy is hardcoded to 8 entities (4+4). Scaling
  to different team sizes or asymmetric matches requires architectural changes.
