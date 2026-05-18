//! ai↔planner bridge — the stable `GrapplePlanner.planGrapple` surface
//! the ported roles/RiggerAI call, expressed on `ai::types`.
//!
//! `crate::planner` (port increment 2) is algorithmically the TS
//! `planGrapple`, but uses planner-local structs. Rather than refactor
//! the shipped/parity-tested planner, this thin orchestrator-authored
//! adapter converts ai::types ↔ planner structs. It is NOT a behavior
//! change: it forwards to `crate::planner::plan_grapple` with the
//! canonical MPC knob (`planner = 0`, = TS `tune('planner', 0)` default).

use super::types::{PlayerSim, SimState, TeamSide};
use crate::math::Vec3;
use crate::planner;
use crate::planner_cost::Profile;
use std::cell::RefCell;
use std::rc::Rc;

/// Re-export of the planner's `Plan` as the AI-facing `GrapplePlan`
/// (TS `GrapplePlan`: anchor_pos, reel ∈ {-1,0}, projected_dist, is_spar).
pub use crate::planner::Plan as GrapplePlan;

/// Active planner algorithm class — the faithful restoration of the old
/// TS `tune('planner', 0)` / `globalThis.__rigtune` knob (0 = MPC,
/// 1 = RRT, 2 = CEM, 3 = MPPI, 4 = SimAnneal, 5 = Beam, 6 = MCTS,
/// 7 = PotentialField, 8 = RandomShooting, 9 = Coordination).
///
/// PRODUCTION DEFAULT = 9 (Coordination). The 8000-tick skill-eval
/// ranked it 44.5 vs MPC 43.5 — the multi-agent cost terms
/// (teammate-interference penalty + pass-setup attractor) measurably
/// beat plain momentum-MPC, and it is the strategic substrate for the
/// cost-term-composition / coordination work to come. The search
/// METHOD was shown not to matter (RandomShooting == MPC); the win is
/// in the composed cost terms, which is what Coordination adds. The
/// in-browser progression gate validates this default behaviorally.
/// The skill-eval harness still flips this per deterministic run.
///
/// Stored THREAD-LOCAL (not a process-global atomic) so the native
/// parallel skill-eval can run independent matches with different
/// classes/profiles on rayon threads without clobbering each other
/// mid-match. Production wasm is single-threaded, so the thread-local
/// default (Coordination, engine 9) is observationally identical to the
/// old global.
///
/// STAGE 2: the knob is generalized from `i32` to `Rc<Profile>`. We hold
/// `Option<Rc<Profile>>`:
///  - `None` ⇒ "use the production default" — resolved lazily, ON FIRST
///    READ, to a single per-thread cached `Rc<Profile::from_class(9)>`.
///  - `Some(p)` ⇒ an explicitly installed profile (eval / future GA).
///
/// Why `RefCell`, not `Cell` (blueprint bug fix): `Cell::get` requires
/// `Copy`, and `Option<Rc<Profile>>` is NOT `Copy` (`Rc` owns a refcount;
/// `Profile` owns a `Vec<Box<dyn CostTerm>>`). `RefCell` borrows soundly
/// with no `Copy` bound. (`Cell::take`/`replace` would also work; `RefCell`
/// is the cleanest read-mostly form here.)
///
/// Why `Rc`, not `Arc`: the cell is `thread_local!` — only ever touched by
/// its owning thread, so no cross-thread sharing exists and the atomic
/// refcount of `Arc` would be pure overhead. `Rc` is wasm32-safe and
/// single-thread-sound here. Both compile for the production wasm cdylib;
/// `Rc` is chosen for being strictly cheaper with identical semantics.
///
/// NO PER-TICK CHURN: `planner_profile()` is called once per `plan_grapple`
/// = per player per 240 Hz tick. The default `Profile::from_class(9)` (which
/// allocates a `Vec<Box<dyn CostTerm>>`) is built AT MOST ONCE PER THREAD:
/// the first read with a `None` cell constructs it, stores the `Rc` back
/// into the cell, and every subsequent read is just an `Rc::clone` (a
/// refcount bump — no `Vec`/`Box` allocation, no `from_class` call). See
/// `planner_profile()` for the lazy-init.
thread_local! {
    static PLANNER_PROFILE: RefCell<Option<Rc<Profile>>> =
        const { RefCell::new(None) };
}

/// Install an explicit planner profile on the calling thread (eval /
/// exploration / future GA). `None` restores the lazily-cached production
/// default (Coordination). Set before a run; constant during it.
/// Thread-local: affects only the calling thread.
pub fn set_planner_profile(profile: Option<Rc<Profile>>) {
    PLANNER_PROFILE.with(|c| *c.borrow_mut() = profile);
}

/// The active planner profile on the calling thread. If none is installed
/// (the production case), returns the per-thread lazily-cached Coordination
/// default (`Profile::from_class(9)`), built at most once per thread and
/// thereafter returned by a cheap `Rc::clone` — NO per-tick allocation.
pub fn planner_profile() -> Rc<Profile> {
    PLANNER_PROFILE.with(|c| {
        // Fast path: already populated (explicit OR the cached default).
        if let Some(p) = c.borrow().as_ref() {
            return Rc::clone(p);
        }
        // Slow path: taken AT MOST ONCE per thread — build the default,
        // cache it back so all future reads are an Rc::clone only.
        let def = Rc::new(Profile::coordination_default());
        *c.borrow_mut() = Some(Rc::clone(&def));
        def
    })
}

/// BACK-COMPAT SHIM. The pre-Stage-2 `i32` knob, preserved verbatim in
/// behavior so existing callers (notably `skill_eval.rs`) are STRUCTURALLY
/// UNTOUCHED — they keep flipping the class and the ranker composites stay
/// provably unchanged. Forwards to `set_planner_profile` with the matching
/// profile.
pub fn set_planner_class(class: i32) {
    set_planner_profile(Some(Rc::new(Profile::from_class(class))));
}

/// Current planner class = the active profile's class number (production
/// default 9 = Coordination). `class_number()` is always the dispatch arm,
/// so this is total; kept for back-compat with i32 callers/tests.
pub fn planner_class() -> i32 {
    planner_profile().class_number()
}

/// TS `sticky?: { pos; reel: -1|0 }` for the anti-dither hysteresis.
#[derive(Clone, Copy, Debug)]
pub struct Sticky {
    pub pos: Vec3,
    pub reel: i32,
}

/// Team → planner's i32 convention (mirrors src/sim/wasm.ts:
/// `team === 'home' ? 0 : 1`). Only distinctness matters — the planner
/// compares `p.team != player.team` exactly as the TS compares the
/// TeamSide strings.
fn team_i32(t: TeamSide) -> i32 {
    match t {
        TeamSide::Home => 0,
        TeamSide::Away => 1,
    }
}

fn to_planner_state(state: &SimState) -> planner::SimState {
    planner::SimState {
        omega: state.omega,
        tick: state.tick as i64,
        players: state
            .players
            .iter()
            .map(|p| planner::PlayerSim {
                id: p.id.clone(),
                team: team_i32(p.team),
                p: p.p,
                v: p.v,
            })
            .collect(),
    }
}

/// 1:1 with TS `planGrapple(player, target, state, avoidDefenders=true,
/// sticky?)` — returns `None` (TS `null`) when already within 3 m.
pub fn plan_grapple(
    player: &PlayerSim,
    target: Vec3,
    state: &SimState,
    avoid_defenders: bool,
    sticky: Option<Sticky>,
) -> Option<GrapplePlan> {
    let pl = planner::PlayerSim {
        id: player.id.clone(),
        team: team_i32(player.team),
        p: player.p,
        v: player.v,
    };
    let st = to_planner_state(state);
    let sk = sticky.map(|s| planner::Sticky {
        pos: s.pos,
        reel: s.reel,
    });
    // STAGE 2: dispatch by the ACTIVE PROFILE's engine (the generalized
    // knob), not a bare i32. Production default = Coordination (engine 9);
    // the skill-eval harness flips the profile (via the i32 shim) to rank
    // the algorithm zoo. `from_class(9).engine == 9` ⇒ identical dispatch
    // arm ⇒ bit-identical Plan.
    let engine = planner_profile().engine;
    planner::plan_grapple(&pl, target, &st, avoid_defenders, sk, engine)
}

/// TS `sparPositions()` — the static spar lattice (axis spine + off-axis
/// rings). Re-exported unchanged.
pub fn spar_positions() -> Vec<Vec3> {
    planner::spar_positions()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::types::{
        AnchorType, BellState, MatchPhase, RiggerRole,
    };
    use crate::math::Quat;

    fn vz() -> Vec3 {
        Vec3 { x: 0.0, y: 0.0, z: 0.0 }
    }

    fn mk_player(id: &str, team: TeamSide, p: Vec3) -> PlayerSim {
        PlayerSim {
            id: id.to_string(),
            team,
            role: RiggerRole::Spinner,
            p,
            v: vz(),
            q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
            line: None,
            dv_budget: 0.0,
            contact_ref: None,
            grounded: false,
        }
    }

    fn mk_state(players: Vec<PlayerSim>) -> SimState {
        SimState {
            tick: 0.0,
            omega: crate::tuning::OMEGA,
            bell: BellState {
                p: vz(),
                v: vz(),
                q: Quat { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                w: vz(),
                chime: 1.0,
                held_by: None,
                thrown_by: None,
                touched_since_throw: false,
                release_pos: vz(),
                release_tick: 0.0,
                pass_chain: vec![],
            },
            players,
        }
    }

    #[test]
    fn close_target_returns_none() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
        let st = mk_state(vec![p.clone()]);
        // within 3 m → TS planGrapple returns null
        assert!(plan_grapple(&p, Vec3 { x: 1.0, y: 0.0, z: 0.0 }, &st, true, None).is_none());
        let _ = (AnchorType::Spar, MatchPhase::Live);
    }

    #[test]
    fn far_target_plans_and_is_deterministic() {
        let p = mk_player("H1", TeamSide::Home, Vec3 { x: -100.0, y: 5.0, z: 0.0 });
        let st = mk_state(vec![
            p.clone(),
            mk_player("A1", TeamSide::Away, Vec3 { x: 50.0, y: 0.0, z: 0.0 }),
        ]);
        let tgt = Vec3 { x: 200.0, y: 0.0, z: 0.0 };
        let a = plan_grapple(&p, tgt, &st, true, None);
        let b = plan_grapple(&p, tgt, &st, true, None);
        assert_eq!(a, b, "bridge must be deterministic");
        if let Some(plan) = a {
            assert!(plan.projected_dist.is_finite());
            assert!(plan.reel == -1 || plan.reel == 0);
        }
    }

    #[test]
    fn spar_lattice_nonempty() {
        assert!(!spar_positions().is_empty());
    }

    // ── STAGE 2: thread-local Profile knob ──────────────────────────────

    /// Default (no profile installed) ⇒ Coordination, engine 9. This is
    /// the production gate: the lazily-cached default is `from_class(9)`.
    #[test]
    fn default_profile_is_coordination_engine_9() {
        // Fresh thread ⇒ untouched thread-local (None ⇒ lazy default).
        std::thread::spawn(|| {
            let p = planner_profile();
            assert_eq!(p.engine, 9, "default engine must be Coordination");
            assert_eq!(p.class_number(), 9);
            assert_eq!(planner_class(), 9, "i32 back-compat default");
        })
        .join()
        .unwrap();
    }

    /// The `set_planner_class` back-compat shim still works: it installs
    /// the matching profile and `planner_class()` round-trips. This is the
    /// path `skill_eval.rs` uses unchanged.
    #[test]
    fn set_planner_class_back_compat() {
        std::thread::spawn(|| {
            set_planner_class(3);
            assert_eq!(planner_class(), 3);
            assert_eq!(planner_profile().engine, 3);
            set_planner_class(9);
            assert_eq!(planner_class(), 9);
            // ≥10 ⇒ MPC-fallback dispatch, engine retains requested id.
            set_planner_class(42);
            assert_eq!(planner_profile().engine, 42);
            assert_eq!(planner_profile().label, "MPC");
        })
        .join()
        .unwrap();
    }

    /// `set_planner_profile` / `planner_profile` round-trip on the calling
    /// thread, and `None` restores the lazily-cached Coordination default.
    #[test]
    fn set_planner_profile_round_trip() {
        std::thread::spawn(|| {
            set_planner_profile(Some(Rc::new(Profile::from_class(5))));
            assert_eq!(planner_profile().engine, 5);
            assert_eq!(planner_class(), 5);
            // None ⇒ back to the production default.
            set_planner_profile(None);
            assert_eq!(planner_profile().engine, 9);
            assert_eq!(planner_class(), 9);
        })
        .join()
        .unwrap();
    }

    /// Thread-local isolation: a profile installed on one thread does NOT
    /// leak into another (mirrors the determinism-test intent — parallel
    /// eval threads must not clobber each other).
    #[test]
    fn profile_is_thread_local_isolated() {
        let t1 = std::thread::spawn(|| {
            set_planner_class(1); // RRT on this thread only
            assert_eq!(planner_class(), 1);
        });
        let t2 = std::thread::spawn(|| {
            // Untouched thread ⇒ still the production default.
            assert_eq!(planner_class(), 9);
            set_planner_class(6);
            assert_eq!(planner_class(), 6);
        });
        t1.join().unwrap();
        t2.join().unwrap();
        // Main thread untouched by either child ⇒ still default.
        assert_eq!(planner_class(), 9);
    }

    /// No per-tick churn: repeated `planner_profile()` reads on the default
    /// path return the SAME cached allocation (same `Rc` data pointer) —
    /// proof the `Vec<Box<dyn CostTerm>>` is built at most once per thread,
    /// never rebuilt per call.
    #[test]
    fn default_profile_not_rebuilt_per_call() {
        std::thread::spawn(|| {
            let a = planner_profile();
            let b = planner_profile();
            let c = planner_profile();
            assert!(Rc::ptr_eq(&a, &b), "default must be cached, not rebuilt");
            assert!(Rc::ptr_eq(&b, &c), "default must be cached, not rebuilt");
        })
        .join()
        .unwrap();
    }
}
