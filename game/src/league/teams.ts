/**
 * RIG / Spindle League — canonical franchise dataset
 *
 * Transcribed from league/teams.md.  Color hex values are documented
 * approximations of the named palette colors:
 *   works-green      #4a7c3f   oxide-orange      #c4622d
 *   gunmetal         #2c3539   safety-yellow     #ffcc00
 *   cream            #fffdd0   maroon            #800020
 *   deep-space black #0d0d0d   telemetry-cyan    #00e5ff
 *   revolutionary-blue #002868  slate            #708090
 *   harvest-gold     #da9100   ivy-green         #4a7c59
 *   chalk-white      #f5f5f0   mourning-violet   #7b4f8e
 *   steel            #71797e   Tyrian purple     #66023c
 *   Lira-gold        #c9a84c   mercenary-grey    #9e9e9e
 *   adriatic-teal    #00827f   rust              #b7410e
 *   binary-white     #f0f0f0   furnace-red       #cc3300
 *   ochre            #cc7722   red-dwarf crimson #8b0000
 *   ash-grey         #b2b2b2   scar-white        #f0ece4
 *   iron             #4a4a4a   storm-violet      #6b3fa0
 *   pale-gold        #d4c17f   twin-amber        #ffbf00
 *   navy             #001f5b   void-black        #0a0a0a
 *   signal-orange    #ff6600   sea-blue          #006994
 *   terracotta       #c06030   barricade-red     #cc1100
 *   worklight-white  #f5f0e8   brown-dwarf maroon #6b2737
 *   dim-amber        #b8860b   hazard-orange     #ff6600
 *   regolith-grey    #9e9587   ice-blue          #a8d8e8
 *   heritage-bronze  #8c6914   oxblood           #6b1c1c
 *   pitch-black      #080808   faint-white       #e8e8e8
 *   pioneer-grey     #7a7a6e   exile-charcoal    #3d3d3d
 *   faded-tricolor   #7a7a9a   reference-grey    #c0c0c0
 */

// ── union types ──────────────────────────────────────────────────────────────

export type Faction = 'FRM' | 'CNE' | 'CON' | 'BLT' | 'EUR' | 'REF' | 'CON/REF' | 'REF/CON';

export type CylinderClass =
  | 'big-slow'   // large agricultural/industrial cylinders — shallow gradient, gentle Coriolis
  | 'small-fast' // Belt/frontier spin-stations — savage gradient, vicious Coriolis
  | 'mid'        // balanced mid cylinders — line/broker game
  | 'neutral';   // negotiated reference spin / no fixed cylinder

export type StyleTag =
  // Fall family — big-slow habitats, patient deep power rig
  | 'fall-dynasty'    // Tuebor: the league archetype, long casts, granite Anchor
  | 'fall-grind'      // Cascade Works: clatter-heavy defensive Fall
  | 'fall-tempo'      // Saginaw Bell: Fall with beautiful clean passing
  | 'fall-loop'       // Veracity Yard: Fall base with Coriolis Loop specialists
  | 'fall-isolated'   // Frontier Veracity Deep: Fall, stubborn, far-edge colony
  // Line family — mid cylinders, Commonwealth core
  | 'line'            // Boston Commons / Plymouth / New Haven / Gliese 65: tether craft, tempo, away-points
  // Ground family — mid habitats, Concordat & CNE
  | 'ground-away'     // Roma Aeterna: broker rig, infuriating away-draws
  | 'ground-broker'   // Tau Ceti Mare Nostrum: perfected ground/line broker
  | 'ground-grief'    // Manchester Memorial / Fomalhaut Dagon Wake: defiant ground under the shadow
  | 'ground-defiant'  // Hebat Resolve: low-scoring, beloved, defiant
  | 'ground-haunted'  // Fomalhaut Dagon Wake: haunted ground under the dead world
  | 'ground-travel'   // Lacaille Longjump: ground, always the away team
  | 'ground-ceremonial' // Vatican Spindle: reference-spin, nobody's home conditions
  | 'adaptive-ground' // Lira Free Company: no home, no home disadvantage
  | 'audio-ground'    // Kuiper Long Call: ground built for audio across enormous gaps
  | 'ground-mournful' // Silesia Exiles: technically gorgeous, perpetually underfunded
  | 'ground-reference' // The Reference Eleven: platonic ground game
  | 'ground-junior'   // 82 Eridani Secondi, Trieste Counterweight: junior-broker / dull-effective
  // Rise family — small-fast habitats, Belt/frontier outposts
  | 'rise-chaos'      // Toliman Gradient: all Freewing, chasing 5s and 7s
  | 'rise-power'      // Centauri b Hammerline: Rise-leaning hammer, wins 7s by force
  | 'rise-exhausted'  // Proxima Long Shift: the long-shift that never ended
  | 'rise-flock'      // Wolf 359 Migration: coordinated high game, whole crews move like a skein
  | 'rise-political'  // Luhman Orbital Riot: labor movement that learned to score
  | 'rise-lowlight'   // Luhman 16 Brownline: low-light audio legends
  | 'rise-lunatic'    // Belt Free Local 9: purest chaos rig
  | 'rise-contrarian' // Ceres Counterspin: attacks Free end by preference
  | 'rise-ceremonial' // Vesta Tool-Bag Originals: tradition as a weapon
  | 'rise-dirty';     // Eros Garrote: foul-prone, magnetic, beloved villains

// ── TeamProfile ──────────────────────────────────────────────────────────────

/**
 * Numeric AI profile derived from style + cylinder physics.
 * All values in [0, 1].
 *
 * freeEndBias      — tendency to attack the Free (antispinward) goal
 * loopPropensity   — tendency to attempt Coriolis Loop (7-pt) plays
 * aggression       — overall forward-press / risk-taking tempo
 * grappleRisk      — tendency toward high-risk line maneuvers
 * contestAggression — likelihood of contesting duels aggressively
 * snatchVsClatter  — 0 = prefers clatter (disrupt); 1 = prefers snatch (steal)
 * awayPointBias    — tendency to play for the away point / road draw
 * variance         — moment-to-moment score variance (chaos vs. consistency)
 */
export interface TeamProfile {
  freeEndBias: number;
  loopPropensity: number;
  aggression: number;
  grappleRisk: number;
  contestAggression: number;
  snatchVsClatter: number;
  awayPointBias: number;
  variance: number;
}

// ── Franchise interface ──────────────────────────────────────────────────────

export interface Franchise {
  id: string;
  name: string;
  faction: Faction;
  conference: 'sol' | 'near' | 'far';
  cylinderClass: CylinderClass;
  /** [primary, secondary] hex color strings */
  colors: [string, string];
  logo: string;
  record: { w: number; l: number };
  /** 1–16 if in the Jump (single-elimination playoff), null otherwise */
  seed: number | null;
  styleTag: StyleTag;
  fanbase: string;
}

// ── All 32 franchises ────────────────────────────────────────────────────────

export const TEAMS: Franchise[] = [
  // ── SOL CONFERENCE — "First Spin" (12) ─────────────────────────────────────

  // FRM Sol bloc — the Fall dynasties
  {
    id: 'tuebor-spin-detroiters',
    name: 'Tuebor Spin Detroiters',
    faction: 'FRM',
    conference: 'sol',
    cylinderClass: 'big-slow',
    colors: ['#4a7c3f', '#c4622d'], // works-green & oxide-orange
    logo: 'Clenched line-claw over the Michigan motto TUEBOR ("I will defend")',
    record: { w: 27, l: 6 },
    seed: 1,
    styleTag: 'fall-dynasty',
    fanbase:
      'Three generations deep, owns the calm. Neutrals are bored by them; Tuebor doesn\'t care, Tuebor spins on.',
  },
  {
    id: 'cascade-works-foremen',
    name: 'Cascade Works Foremen',
    faction: 'FRM',
    conference: 'sol',
    cylinderClass: 'big-slow',
    colors: ['#2c3539', '#ffcc00'], // gunmetal & safety-yellow
    logo: 'Torque wrench bent into a ring',
    record: { w: 21, l: 12 },
    seed: 11,
    styleTag: 'fall-grind',
    fanbase: 'Actual riggers and yard crews; the loudest rim-band seats in the league.',
  },
  {
    id: 'saginaw-bell',
    name: 'Saginaw Bell',
    faction: 'FRM',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#fffdd0', '#800020'], // cream & maroon
    logo: 'Tuning-fork bell',
    record: { w: 18, l: 15 },
    seed: null,
    styleTag: 'fall-tempo',
    fanbase:
      'The connoisseur\'s team. Bell rings true more than any club; scores least dramatically.',
  },
  {
    id: 'veracity-yard-probes',
    name: 'Veracity Yard Probes',
    faction: 'FRM',
    conference: 'sol',
    cylinderClass: 'big-slow',
    colors: ['#0d0d0d', '#00e5ff'], // deep-space black & telemetry-cyan
    logo: 'Probe silhouette on a parabolic line',
    record: { w: 16, l: 17 },
    seed: null,
    styleTag: 'fall-loop',
    fanbase:
      'Shipwrights, the FASTPITCH faithful, gallows humor. Lost the play-in on a clattered bell.',
  },

  // CNE Sol bloc — the line game
  {
    id: 'boston-commons-reach',
    name: 'Boston Commons Reach',
    faction: 'CNE',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#002868', '#fffdd0'], // revolutionary-blue & cream
    logo: 'Goal ring over a cobblestone field',
    record: { w: 24, l: 9 },
    seed: 2,
    styleTag: 'line',
    fanbase:
      'The Commonwealth\'s pride; FRM\'s friendly twin-league nemesis. The great Tuebor-vs-Commons rivalry.',
  },
  {
    id: 'plymouth-counter',
    name: 'Plymouth Counter',
    faction: 'CNE',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#708090', '#da9100'], // slate & harvest-gold
    logo: 'Counterweight on a plumb line',
    record: { w: 20, l: 13 },
    seed: null,
    styleTag: 'line',
    fanbase: 'Old Commonwealth money, very smug, deservedly.',
  },
  {
    id: 'new-haven-ear',
    name: 'New Haven Ear',
    faction: 'CNE',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#4a7c59', '#f5f5f0'], // ivy-green & chalk-white
    logo: 'Blindfolded ear inside a ring',
    record: { w: 19, l: 14 },
    seed: null,
    styleTag: 'line',
    fanbase:
      'Home of the officiating academy. They win the Ear\'s calls because they literally train the Ears. Everyone hates this.',
  },
  {
    id: 'manchester-memorial',
    name: 'Manchester Memorial',
    faction: 'CNE',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#7b4f8e', '#71797e'], // mourning-violet & steel
    logo: 'Broken ring mended with a line',
    record: { w: 22, l: 11 },
    seed: 5,
    styleTag: 'ground-grief',
    fanbase:
      'The league\'s emotional heart. Every match opens with a silent bell. When Memorial makes a run, even Frontier crowds stand.',
  },

  // Concordat Sol bloc — the brokers
  {
    id: 'roma-aeterna-brokers',
    name: 'Roma Aeterna Brokers',
    faction: 'CON',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#66023c', '#c9a84c'], // Tyrian purple & Lira-gold
    logo: 'She-wolf curled into a ring, Vatican key crossed with a line',
    record: { w: 23, l: 10 },
    seed: 3,
    styleTag: 'ground-away',
    fanbase:
      'Cosmopolitan, multilingual, shows up everywhere. Quietly the most-followed club across non-aligned space.',
  },
  {
    id: 'vatican-spindle',
    name: 'Vatican Spindle',
    faction: 'CON/REF',
    conference: 'sol',
    cylinderClass: 'neutral',
    colors: ['#ffffff', '#ffd700'], // white & gold
    logo: 'Crossed keys over the reference arc',
    record: { w: 17, l: 16 },
    seed: null,
    styleTag: 'ground-ceremonial',
    fanbase:
      'Exists more as institution than contender. The joke writes itself; the joke is canon; the joke is load-bearing.',
  },
  {
    id: 'lira-free-company',
    name: 'Lira Free Company',
    faction: 'CON',
    conference: 'sol',
    cylinderClass: 'neutral',
    colors: ['#9e9e9e', '#c9a84c'], // mercenary-grey & gold
    logo: 'Coin on a line, no home ring',
    record: { w: 20, l: 13 },
    seed: 14,
    styleTag: 'adaptive-ground',
    fanbase:
      'Belt operators and frontier outposts adopt them as a flag of convenience. A genuinely strange, beloved team.',
  },
  {
    id: 'trieste-counterweight',
    name: 'Trieste Counterweight',
    faction: 'CON',
    conference: 'sol',
    cylinderClass: 'mid',
    colors: ['#00827f', '#b7410e'], // adriatic-teal & rust
    logo: 'Balance scale as two rings',
    record: { w: 15, l: 18 },
    seed: null,
    styleTag: 'ground-junior',
    fanbase: 'The Concordat\'s farm of brokers.',
  },

  // ── THE NEAR REACH — "First Light" (10) ─────────────────────────────────────

  {
    id: 'toliman-gradient',
    name: 'Toliman Gradient',
    faction: 'FRM',
    conference: 'near',
    cylinderClass: 'small-fast',
    colors: ['#f0f0f0', '#cc3300'], // binary-white & furnace-red
    logo: 'Two suns through a ring with a Coriolis spiral tail',
    record: { w: 25, l: 8 },
    seed: 8,
    styleTag: 'rise-chaos',
    fanbase:
      'Feral, sleepless (binary day), evangelical. The franchise that proved the colonies could beat Sol.',
  },
  {
    id: 'centauri-b-hammerline',
    name: 'Centauri b Hammerline',
    faction: 'FRM',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#cc7722', '#1a1a1a'], // ochre & black
    logo: 'Hammer whose head is a line-claw',
    record: { w: 21, l: 12 },
    seed: 10,
    styleTag: 'rise-power',
    fanbase: 'The colony that out-produces the Concordat; the team plays like it knows that.',
  },
  {
    id: 'proxima-long-shift',
    name: 'Proxima Long Shift',
    faction: 'FRM',
    conference: 'near',
    cylinderClass: 'small-fast',
    colors: ['#8b0000', '#1a0a0a'], // red-dwarf crimson & dark
    logo: 'Single dim sun, a long line',
    record: { w: 17, l: 16 },
    seed: null,
    styleTag: 'rise-exhausted',
    fanbase: 'Frontier romance incarnate. Plays like the work shift that invented the game never ended.',
  },
  {
    id: 'fomalhaut-dagon-wake',
    name: 'Fomalhaut Dagon Wake',
    faction: 'CNE',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#b2b2b2', '#8b0000'], // ash-grey & deep-red
    logo: 'Cracked planet behind a ring',
    record: { w: 19, l: 14 },
    seed: null,
    styleTag: 'ground-haunted',
    fanbase:
      'Plays under the knowledge of what a relativistic impactor did to a living world. NISO does not love their iconography.',
  },
  {
    id: 'hebat-resolve',
    name: 'Hebat Resolve',
    faction: 'CNE',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#f0ece4', '#4a4a4a'], // scar-white & iron
    logo: 'Ring scored by a line that didn\'t break',
    record: { w: 22, l: 11 },
    seed: 9,
    styleTag: 'ground-defiant',
    fanbase:
      'Sister-grief franchise to Manchester Memorial; their meetings are the most-listened rig in human space.',
  },
  {
    id: 'wolf-359-migration',
    name: 'Wolf 359 Migration',
    faction: 'CNE',
    conference: 'near',
    cylinderClass: 'small-fast',
    colors: ['#6b3fa0', '#d4c17f'], // storm-violet & pale-gold
    logo: 'Skein of migrating birds bent into a Coriolis arc',
    record: { w: 20, l: 13 },
    seed: 15,
    styleTag: 'rise-flock',
    fanbase:
      'Obsessed with the macrofauna question. The team leans all the way into it. CNE finds this useful and unsettling.',
  },
  {
    id: 'gliese-65-twinstar',
    name: 'Gliese 65 Twinstar',
    faction: 'CNE',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#ffbf00', '#001f5b'], // twin-amber & navy
    logo: 'Two rings interlocked',
    record: { w: 18, l: 15 },
    seed: null,
    styleTag: 'line',
    fanbase: 'Binary-system colony, balanced line game.',
  },
  {
    id: 'lacaille-longjump',
    name: 'Lacaille Longjump',
    faction: 'CNE',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#0a0a0a', '#ff6600'], // void-black & signal-orange
    logo: 'Dotted jump-arc through a ring',
    record: { w: 16, l: 17 },
    seed: null,
    styleTag: 'ground-travel',
    fanbase: 'Always the away team. Travel-hardened. Named for the jump that reaches them.',
  },
  {
    id: 'tau-ceti-mare-nostrum',
    name: 'Tau Ceti Mare Nostrum',
    faction: 'CON',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#006994', '#c9a84c'], // sea-blue & gold
    logo: 'Wave curling into a ring, MARE NOSTRUM',
    record: { w: 24, l: 9 },
    seed: 4,
    styleTag: 'ground-broker',
    fanbase:
      'Plays for the away point and the draw and wins the Jump doing it. The Concordat\'s whole foreign policy as a sport.',
  },
  {
    id: '82-eridani-secondi',
    name: '82 Eridani Secondi',
    faction: 'CON',
    conference: 'near',
    cylinderClass: 'mid',
    colors: ['#c06030', '#c9a84c'], // terracotta & gold
    logo: 'Small ring inside a larger one',
    record: { w: 15, l: 18 },
    seed: null,
    styleTag: 'ground-junior',
    fanbase: 'The Concordat\'s far farm.',
  },

  // ── THE FAR REACH & THE BELT — "Deep Jump" (10) ─────────────────────────────

  {
    id: 'luhman-orbital-riot',
    name: 'Luhman Orbital Riot',
    faction: 'FRM',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#cc1100', '#f5f0e8'], // barricade-red & worklight-white
    logo: 'Raised line-claw, cracked ring',
    record: { w: 23, l: 10 },
    seed: 6,
    styleTag: 'rise-political',
    fanbase:
      'A franchise born from the France-backed workers\' riot. A labor movement that learned to score. France-watchers watch this fanbase. So does Ananke.',
  },
  {
    id: 'luhman-16-brownline',
    name: 'Luhman 16 Brownline',
    faction: 'FRM',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#6b2737', '#b8860b'], // brown-dwarf maroon & dim-amber
    logo: 'Faint line in the dark',
    record: { w: 17, l: 16 },
    seed: null,
    styleTag: 'rise-lowlight',
    fanbase: 'Low-light audio legends — they broadcast better than they look.',
  },
  {
    id: 'belt-free-local-9',
    name: 'Belt Free Local 9',
    faction: 'BLT',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#ff6600', '#1a1a1a'], // hazard-orange & black
    logo: 'Worn tool-bag on a frayed line',
    record: { w: 21, l: 12 },
    seed: 12,
    styleTag: 'rise-lunatic',
    fanbase:
      'The dual-use talent fountain, made visible. Half the league\'s best riggers came up through Belt Locals — and so did some people the league does not talk about.',
  },
  {
    id: 'ceres-counterspin',
    name: 'Ceres Counterspin',
    faction: 'BLT',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#9e9587', '#a8d8e8'], // regolith-grey & ice-blue
    logo: 'Station ring spinning the wrong way',
    record: { w: 20, l: 13 },
    seed: 13,
    styleTag: 'rise-contrarian',
    fanbase: 'Attacks the Free end by preference, the most expensive way to play, because Belters are like that.',
  },
  {
    id: 'vesta-tool-bag-originals',
    name: 'Vesta Tool-Bag Originals',
    faction: 'BLT',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#8c6914', '#1a1a1a'], // heritage-bronze & black
    logo: 'The tool-bag, gilded, in a ring — "WE THREW THE FIRST LOOP"',
    record: { w: 18, l: 15 },
    seed: 16,
    styleTag: 'rise-ceremonial',
    fanbase:
      'Every origin-myth argument in the sport routes through Vesta\'s marketing department.',
  },
  {
    id: 'eros-garrote',
    name: 'Eros Garrote',
    faction: 'BLT',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#6b1c1c', '#ffd700'], // oxblood & yellow
    logo: 'Line drawn taut across a ring (the named foul, worn on purpose)',
    record: { w: 19, l: 14 },
    seed: null,
    styleTag: 'rise-dirty',
    fanbase:
      'The league\'s beloved villains. Best road-call ratings in the league; villains travel well on audio.',
  },
  {
    id: 'kuiper-long-call',
    name: 'Kuiper Long Call',
    faction: 'BLT',
    conference: 'far',
    cylinderClass: 'small-fast',
    colors: ['#080808', '#e8e8e8'], // pitch-black & faint-white
    logo: 'Single sound-ring expanding',
    record: { w: 22, l: 11 },
    seed: 7,
    styleTag: 'audio-ground',
    fanbase:
      'Their re-call is the single highest-rated media product in human space. The richest "small" team in the league.',
  },
  {
    id: 'frontier-veracity-deep',
    name: 'Frontier Veracity Deep',
    faction: 'FRM',
    conference: 'far',
    cylinderClass: 'big-slow',
    colors: ['#7a7a6e', '#3a6b40'], // pioneer-grey & green
    logo: 'Frontier stake as a goal post',
    record: { w: 17, l: 16 },
    seed: null,
    styleTag: 'fall-isolated',
    fanbase: 'The far edge of the Centauri push. Fall, isolated, stubborn.',
  },
  {
    id: 'silesia-exiles',
    name: 'Silesia Exiles',
    faction: 'EUR',
    conference: 'far',
    cylinderClass: 'mid',
    colors: ['#3d3d3d', '#7a7a9a'], // exile-charcoal & faded-tricolor
    logo: 'Closed ring with a small open break — a door left ajar',
    record: { w: 19, l: 14 },
    seed: null,
    styleTag: 'ground-mournful',
    fanbase:
      'The neutral\'s neutral favorite. Old Europe\'s only seat at the interstellar table. A Concordat-adjacent storyline soaked in ASHFIELD history.',
  },
  {
    id: 'the-reference-eleven',
    name: 'The Reference Eleven',
    faction: 'REF/CON',
    conference: 'far',
    cylinderClass: 'neutral',
    colors: ['#ffffff', '#c0c0c0'], // pure white & reference-grey
    logo: 'The bare reference arc, no team mark',
    record: { w: 0, l: 0 }, // exhibition; no competitive record
    seed: null,
    styleTag: 'ground-reference',
    fanbase:
      'No fans by birth, only by belief. Exists to prove the Reg is real. The sport\'s metric system, wearing a jersey.',
  },
];

// ── Jump seedings ────────────────────────────────────────────────────────────

/**
 * The 16 teams in the Jump (interstellar single-elimination playoff),
 * sorted by seed 1..16 exactly per teams.md §"The Jump — Current Playoff
 * Seeding" table.
 */
export const JUMP_SEEDS: Franchise[] = TEAMS.filter((t) => t.seed !== null).sort(
  (a, b) => (a.seed as number) - (b.seed as number),
);

// ── styleToProfile ───────────────────────────────────────────────────────────

/**
 * Returns a numeric TeamProfile for a given style + cylinder class.
 *
 * Derived from the physics → style → soul canon mapping in sport.md §6:
 *   big-slow  → Fall: low freeEndBias/loop, heavy aggression at depth,
 *               low variance (patient dynasties)
 *   small-fast → Rise: high freeEndBias/loop/variance, high grappleRisk
 *               (legends and heartbreak)
 *   mid / neutral → Ground/Line: high awayPointBias/snatch, moderate
 *               aggression, low variance (broker rig)
 *
 * StyleTag further shifts individual dimensions within those baselines.
 */
export function styleToProfile(styleTag: StyleTag, cylinderClass: CylinderClass): TeamProfile {
  // ── cylinder-class baselines ─────────────────────────────────────────────
  let freeEndBias = 0.5;
  let loopPropensity = 0.5;
  let aggression = 0.5;
  let grappleRisk = 0.5;
  let contestAggression = 0.5;
  let snatchVsClatter = 0.5;
  let awayPointBias = 0.5;
  let variance = 0.5;

  switch (cylinderClass) {
    case 'big-slow':
      // Fall: patient, deep, Faithful-end power
      freeEndBias = 0.2;
      loopPropensity = 0.15;
      aggression = 0.6;
      grappleRisk = 0.3;
      contestAggression = 0.65;
      snatchVsClatter = 0.35;
      awayPointBias = 0.3;
      variance = 0.2;
      break;
    case 'small-fast':
      // Rise: high-free, all Freewing, chasing 5s and 7s
      freeEndBias = 0.8;
      loopPropensity = 0.75;
      aggression = 0.75;
      grappleRisk = 0.8;
      contestAggression = 0.7;
      snatchVsClatter = 0.65;
      awayPointBias = 0.4;
      variance = 0.8;
      break;
    case 'mid':
      // Ground/line: broker rig, tempo, away draws
      freeEndBias = 0.4;
      loopPropensity = 0.25;
      aggression = 0.4;
      grappleRisk = 0.35;
      contestAggression = 0.45;
      snatchVsClatter = 0.7;
      awayPointBias = 0.65;
      variance = 0.3;
      break;
    case 'neutral':
      // Adaptive / reference: all-balanced, no home-field shape
      freeEndBias = 0.5;
      loopPropensity = 0.35;
      aggression = 0.4;
      grappleRisk = 0.35;
      contestAggression = 0.45;
      snatchVsClatter = 0.6;
      awayPointBias = 0.6;
      variance = 0.25;
      break;
  }

  // ── style-tag adjustments (±delta on top of baseline) ───────────────────
  switch (styleTag) {
    case 'fall-dynasty':
      // Granite Anchor, long casts — dial up deep contestAggression, dial down free
      aggression += 0.1;
      contestAggression += 0.1;
      variance -= 0.05;
      break;
    case 'fall-grind':
      // Clatter-heavy defense
      snatchVsClatter -= 0.2;
      aggression -= 0.05;
      variance -= 0.05;
      break;
    case 'fall-tempo':
      // Beautiful clean passing
      snatchVsClatter += 0.1;
      awayPointBias += 0.1;
      variance += 0.05;
      break;
    case 'fall-loop':
      // Engineers doing Coriolis math for fun
      loopPropensity += 0.2;
      grappleRisk += 0.1;
      variance += 0.1;
      break;
    case 'fall-isolated':
      // Stubborn, far-edge
      aggression -= 0.05;
      variance -= 0.05;
      break;
    case 'line':
      // Tether craft, tempo, away-points
      awayPointBias += 0.1;
      grappleRisk += 0.1;
      snatchVsClatter += 0.05;
      break;
    case 'ground-away':
      // Infuriating broker rig, wins on road
      awayPointBias += 0.15;
      aggression -= 0.1;
      variance -= 0.05;
      break;
    case 'ground-broker':
      // Perfected draw game
      awayPointBias += 0.2;
      snatchVsClatter += 0.1;
      aggression -= 0.1;
      break;
    case 'ground-grief':
    case 'ground-haunted':
      // Defiant, low-scoring
      aggression -= 0.05;
      variance -= 0.05;
      freeEndBias -= 0.05;
      break;
    case 'ground-defiant':
      // Low-scoring, beloved
      aggression -= 0.05;
      variance -= 0.05;
      break;
    case 'ground-travel':
      // Always away — lean into the away-point game
      awayPointBias += 0.15;
      variance += 0.05;
      break;
    case 'ground-ceremonial':
    case 'ground-reference':
      // Reference spin, no home-advantage skew
      variance -= 0.1;
      awayPointBias += 0.05;
      break;
    case 'adaptive-ground':
      // No home, no home disadvantage
      awayPointBias += 0.1;
      variance += 0.05;
      break;
    case 'audio-ground':
      // Slow, hypnotic; built for audio
      aggression -= 0.1;
      variance -= 0.05;
      awayPointBias += 0.1;
      break;
    case 'ground-mournful':
      // Technically gorgeous, underfunded
      snatchVsClatter += 0.1;
      variance -= 0.05;
      break;
    case 'ground-junior':
      // Junior-broker / dull-effective
      aggression -= 0.1;
      variance -= 0.1;
      break;
    case 'rise-chaos':
      // All Freewing, chasing 5s and 7s
      freeEndBias += 0.1;
      loopPropensity += 0.1;
      variance += 0.1;
      break;
    case 'rise-power':
      // Wins 7s by force
      aggression += 0.1;
      grappleRisk += 0.05;
      freeEndBias += 0.05;
      break;
    case 'rise-exhausted':
      // Magnificent but tired
      variance += 0.05;
      aggression -= 0.05;
      break;
    case 'rise-flock':
      // Coordinated high game
      contestAggression += 0.1;
      snatchVsClatter += 0.1;
      variance -= 0.05;
      break;
    case 'rise-political':
      // Furious, electric
      aggression += 0.1;
      grappleRisk += 0.1;
      variance += 0.05;
      break;
    case 'rise-lowlight':
      // Audio legends
      variance += 0.05;
      awayPointBias += 0.05;
      break;
    case 'rise-lunatic':
      // Purest chaos rig
      variance += 0.1;
      freeEndBias += 0.05;
      grappleRisk += 0.1;
      break;
    case 'rise-contrarian':
      // Attacks Free end by preference
      freeEndBias += 0.1;
      grappleRisk += 0.05;
      aggression += 0.05;
      break;
    case 'rise-ceremonial':
      // Tradition as a weapon
      loopPropensity += 0.1;
      variance -= 0.05;
      break;
    case 'rise-dirty':
      // Foul-prone, magnetic
      grappleRisk += 0.1;
      aggression += 0.1;
      contestAggression += 0.05;
      variance += 0.05;
      break;
  }

  // clamp all to [0, 1]
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  return {
    freeEndBias: clamp(freeEndBias),
    loopPropensity: clamp(loopPropensity),
    aggression: clamp(aggression),
    grappleRisk: clamp(grappleRisk),
    contestAggression: clamp(contestAggression),
    snatchVsClatter: clamp(snatchVsClatter),
    awayPointBias: clamp(awayPointBias),
    variance: clamp(variance),
  };
}
