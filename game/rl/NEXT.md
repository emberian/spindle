# RIG RL — Future Prospects & Plans

## Immediate next steps (high-leverage, well-scoped)

### 1. Analytical backprop for the critic
The centralized value is a 2-layer MLP (137→128→128→1, tanh). Its backward pass
is textbook: `dL/dW = dL/dout · dout/dactivation · dactivation/dW` through two
tanh layers. This alone would make value-function learning ~50x more sample-efficient
(the value loss is currently the noisiest signal because ES over 34k params is near-random).

**Files:** `rig-core/src/rl/value.rs` — add `backward(loss_grad) → weight_grads`  
**Effort:** ~100 LOC, well-tested against finite-difference  
**Impact:** Value loss converges in 10 gens instead of 100 → better GAE → better policy gradient

### 2. Analytical backprop for the actor (attention)
Harder but highest-impact. The backward through:
- Linear layers (trivial)
- tanh (trivial: `1 - tanh²`)
- Multi-head attention (softmax backward + the Q/K/V projections)
- Residual connections (grad passthrough)

**Files:** `rig-core/src/rl/attention.rs` — add `backward(loss_grad) → weight_grads`  
**Effort:** ~200-300 LOC  
**Impact:** True PPO at 24k params with proper gradient signal. The difference between
"working prototype" and "emergent coordination."

### 3. Reweight the shaping potential
The pass-chain term in Phi is trivially reachable (pass among teammates forever = keep-away).
Rebalance: gate-progress should dominate pass-chain in Phi so the dense gradient points
toward advancing the cast, not just maintaining possession.

**Files:** `rig-core/src/gym.rs` (the Phi computation in `step`)  
**Effort:** ~10 LOC tuning  
**Impact:** May be enough to escape the keep-away basin even without backprop

### 4. Checkpoint serialization + resume
Write the full training state (actor/critic weights, population, rng state, generation)
to disk every N gens. Resume from checkpoint on restart.

**Files:** `rig-core/src/rl/mappo.rs`  
**Effort:** ~50 LOC (serde the MappoState to bincode/JSON)  
**Impact:** Enables long overnight runs without losing progress to crashes

## Medium-term (the "massive arena" setup)

### 5. PyTorch/JAX training backend
Use the Python FFI bridge (already built, 8250 t/s) as the rollout engine for a proper
gradient-based training framework:
- Rust gym provides deterministic environments (fast, parallel, GIL-released)
- Python handles the policy network (PyTorch attention model), PPO update (with real autograd),
  population management, logging/wandb, distributed rollout coordination
- Trained weights serialize back to the Rust forward-pass format for browser deployment

This is the standard "fast sim in compiled language + ML framework for training" split
(see MuJoCo+PyTorch, Isaac Gym, etc). The bridge is already done.

### 6. Distributed rollout workers
For truly massive runs: multiple machines each running VecRigGym(32+), collecting trajectories,
shipping them to a central learner. The gym's `snapshot/restore` makes this clean — workers
can checkpoint partial trajectories and resume deterministically.

### 7. League structure (AlphaStar-style)
Evolve the flat population into a structured league:
- **Main agents**: trained against the full league distribution
- **Main exploiters**: trained specifically to beat the current main agent (finds weaknesses)
- **League exploiters**: trained to beat any agent in the league (generalists)
- Matchmaking by Trueskill/Elo rating matrix

### 8. Curriculum learning
Start with dense reward (direct gate-progress) and anneal toward pure-intrinsic over
training. The policy first learns "how to advance" with a strong gradient, then refines
under the sparse signal that actually reflects the sport's outcome. The Phi potential
already supports this — just schedule `shaping_gamma` from 0.999 toward 0 over training.

## Long-term vision

### 9. Emergent communication
Add a small discrete communication channel (a 4-bit message each agent can emit per tick,
visible to teammates). Let the learner discover coordination protocols — "I'm going for the
bell" / "I'm open" / "switch marks" should emerge from the team reward pressure. The
attention over teammate entities is the natural place to consume these messages.

### 10. Heterogeneous roles from learning
Currently the attention policy has shared weights — all 4 agents are identical.
An extension: condition the policy on a "role embedding" that's part of the learned
parameters. The Director assigns role-indices; the policy's behavior is role-conditioned.
Role differentiation emerges from the team reward rather than being hand-coded.

### 11. Transfer across cylinder configs
The game has multiple cylinder configurations (big-slow, small-fast, mid). A truly
robust policy should generalize across them (different ω, R, L → different Coriolis,
different strategy). Train on a distribution of configs; the physics vary but the
coordination problem is the same class.

### 12. Human-AI teaming
The ultimate coordination test: a learned policy that can coordinate with a HUMAN
teammate (not just copies of itself). The human uses the existing input system
(RMB grapple, F catch, throw); 3 AI riggers must learn to complement the human's
unpredictable behavior. The gym already supports mixed control (some controlled, some AI).

## What makes this project valuable/unique

- **Bit-exact deterministic sim** — not approximate; exact replay, exact counterfactuals
- **Browser-deployable inference** — trained attention policy runs in wasm, watchable live
- **The game IS the visualizer** — no separate demo; what you watch is what trained
- **Real adversarial sport with structure** — not a toy gridworld; Coriolis physics,
  structured scoring (gate/cast progression), contested possession, team roles
- **The sport is canon** (Kanzo's Vivere Astra campaign) — it has culture, a league,
  32 teams, a bracket. The RL is embedded in something with narrative weight
- **Human+AI co-development** — the project itself is a live experiment in building
  complex systems with AI collaboration, documented honestly in the explainer
