# Spindle — Research Synthesis (Phase 1)

Sources swept via Kagi (raw JSON in `research/raw/`), depth-fetched: the
*npj Microgravity* trajectory paper (PMC10570307), Oikofuge's Coriolis
guides, Wikipedia O'Neill cylinder, Wikipedia 17776.

---

## A. The arena physics — this *is* the sport

**Canonical O'Neill cylinder.** ~6.4–8 km diameter (radius R ≈ 3.2–4 km),
32 km long, ~28 rotations/hour (ω ≈ 0.049 rad/s, period ≈ 128 s), 1 g at
the rim. Counter-rotating pair, three land stripes alternating with three
window stripes, ~tens of millions of people. **The central axis is a
zero-gravity region O'Neill himself earmarked for recreation.** We are
not inventing the venue — the canon hands it to us.

**The gravity gradient is the play space, not a backdrop.** Effective
gravity scales with radius: 1 g at the rim → 0 g at the axis. The sport
lives in the long weightless **axial volume** (working name: *the
spindle* / *the calm*), but the gradient *beneath* it is a strategic
vertical axis. Going "down" toward the rim = heavier, faster, more
committed, harder to change your mind. Going "up" toward the axis =
floatier, freer, slower, more reversible. **Radius is a resource a
player spends.**

**Coriolis is the signature trick, and it is asymmetric.** From the
trajectory paper + Oikofuge:
- Thrown objects trace **roulette curves** — involutes (for "dropped"
  objects) and Archimedean spirals (for paths crossing the axis). Most
  throws are hybrids.
- In a habitat spinning one way, balls always curve the same way
  (say, rightward).
- **Throwing spinward (with the spin):** fairly normal-looking arc.
- **Throwing antispinward (against the spin):** the ball *lifts*,
  curves hard, and can circle a long way before coming down. This is
  not intuitive to anyone gravity-raised and it is the sport's beating
  heart.
- **The throw–catch loop:** throw up-and-slightly-antispinward at the
  right speed and the ball loops back to your own hand. A closed orbit
  you can only do here. Free trick shot built into the universe.
- Rule of thumb for a designer: deflection grows with **flight time**
  and with **how much of the spinward/antispinward axis the path
  crosses**; throws *parallel to the cylinder axis* barely deflect.
- Near the rim of a *km-scale* cylinder, Coriolis is nearly invisible
  at human scale (huge R). It becomes dramatic **as you approach the
  axis** and for **long flights** — i.e. precisely in the spindle.
- Tangential speed of the rim is enormous (ωR ≈ 150+ m/s). Anything
  launched off the rotating floor toward the axis **carries that
  sideways momentum** and spirals unless it sheds it. Launching into
  the calm is a skill: you must *shed spin*.

**Sim-fidelity equation (for the playable match later).** In the
rotating frame, point-to-point flight:
`T(t) = e^(−iωt)·[ (t/τ)·e^(iωτ)·(Ω₂−Ω₁) + Ω₁ ]`, ω = habitat angular
velocity, Ω₁/Ω₂ = release/catch positions (complex, plane ⟂ axis), τ =
flight time. Motion parallel to the axis is just inertial (drift +
nothing). This is enough to drive a faithful, *fun-because-real* ball
model.

---

## B. How a sport is actually founded & spread (the arc template)

- **Naismith / basketball:** invented *deliberately*, in a *constrained
  space*, with *improvised equipment* (a peach basket), as **13 rules**,
  to burn energy and build community indoors in winter. Lesson: sports
  are born from a constraint + junk + a bored, energetic population.
- **Association football:** folk games → **public-school codification**
  (rules written so strangers from different schools could play each
  other) → governing body (FIFA, 1930) → global. The codification step
  exists *specifically* to let outsiders play together. Direct analogue
  to standardizing across cylinders of different radius and spin.
- **Folk/pickup research:** throwing-at-a-target games are a *human
  universal* (hoop-and-pole, chunkey, snow snake); pickup games
  self-organize, need almost no gear, and teach cooperation. Whatever
  the sport is, its *seed form* must be playable with what a habitat
  crew literally has lying around.
- **Why new sports spread now (padel, pickleball, teqball, drone
  racing, sepak takraw):** low barrier to entry, social, **legible to
  spectators**, adaptable to available space. These are the levers.
- **Real space-sport precedent:** ISS crews already improvise — floating
  "space soccer," zero-g ping-pong, gymnastics; the Space Games
  Federation is deliberately designing 0-g sport and coined the
  *Astrolete*. Real history says: improvised adaptation first, then
  deliberate design, then a federation.

**Therefore the founding arc is locked:**
construction crews ("riggers") improvising during cylinder build-out →
inter-crew folk game → first written rules so different crews/cylinders
can play each other → a single regulation-arena spec (because every real
cylinder differs) → interstellar league once the O-drive makes travel
feasible. Codification is *driven by the physics problem* that no two
habitats spin alike.

---

## C. The 17776 spirit (tone & structure to borrow)

Premise: immortal humanity keeps playing football at absurd scales,
witnessed by sentient space probes; multimedia; melancholy philosophy
braided with gleeful immaturity. Takeaway for us: **constraint-breaking
deepens emotional resonance** — the *why people play* carries the piece
as much as the rules. We borrow: a witnessing-narrator frame, the sport
as a vessel for a scattered people's identity, and a **baseball soul
(pastoral, clock-less, stat-haunted, individual duels) in a football
body (territory, possession, set pieces, scoring zones).** Most star
systems were colonized by ex-Americans — the game *should* read as the
bastard child of the two American religions, reinvented for vacuum.

---

## D. Locked design constraints carried into Phase 2

1. **Venue:** the weightless axial spindle of an O'Neill cylinder; the
   rim→axis gravity gradient is an in-play strategic dimension, not set
   dressing.
2. **Signature mechanics to exploit:** Coriolis asymmetry (antispinward
   = lift/curve), the throw–catch loop, spin-shedding on launch,
   "spending radius," roulette ball paths.
3. **Equipment is improvisable from build materials.** Players can't fly
   freely in 0 g — you change your vector by pushing off structure or by
   **tether/grapple**. The hook is *physically mandatory*, not a gimmick:
   this is where the user's grapple/arena-shooter/Xonotic instinct is
   not only allowed but demanded by the physics.
4. **Football DNA:** territory, possession with a downs-like structure,
   set pieces, scoring zones. **Baseball DNA:** discrete one-on-one
   duels, no game clock, deep idiosyncratic statistics, pastoral
   timelessness.
5. **Founding arc:** riggers' improvised game → codified for inter-
   cylinder play → regulation-arena standardization → interstellar
   league (arc detailed in Phase 2).
6. **The sport is a culture-carrier.** Each habitat's physical
   parameters (radius, spin rate, length, who built it, why) shape its
   style of play and its people's temperament — the spine that connects
   Phase 2 (design) to Phase 4 (the 32+ teams) to Vivere Astra's
   factions (FRM / CNE / Concordat / Belt / France).

---

## E. Key sources (for the PDF bibliography)

- *npj Microgravity* — "Simplified equations for object trajectories in
  rotating space habitats" (PMC10570307) — the ball physics.
- Oikofuge — "Coriolis Effect In A Rotating Space Habitat" (+ axial
  supplement) — the intuition.
- Wikipedia — *O'Neill cylinder*; NSS O'Neill settlement page — venue.
- Jon Bois — *17776 / What Football Will Look Like in the Future*
  (SB Nation) — tone & structural inspiration.
- Histories: basketball (Naismith), association football codification,
  pickup/folk games, Space Games Federation / ISS improvised sport.
