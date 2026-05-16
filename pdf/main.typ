// RIG — A Vivere Astra worldbuilding dossier
// Build: typst compile main.typ rig.pdf

#let ink    = rgb("#11131a")
#let paper  = rgb("#f4f1ea")
#let cyan   = rgb("#1aa6b7")   // telemetry / the calm
#let oxide  = rgb("#d4602a")   // works-orange / the Rise
#let dim    = rgb("#6b7079")
#let gold   = rgb("#c9a24a")

#set document(title: "RIG — The Game in the Calm", author: "Spindle dossier")
#set text(font: "New Computer Modern", size: 10.5pt, fill: ink)
#set par(justify: true, leading: 0.68em)
#set page(fill: paper, margin: (x: 2.6cm, y: 2.4cm))

// ---- Archimedean spiral motif (built-in only) ----
#let spiral(turns: 3.0, scale: 1.0, stroke-col: cyan, w: 0.8pt) = {
  let pts = ()
  let n = 240
  for i in range(n + 1) {
    let t = i / n * turns * 2 * calc.pi
    let r = (i / n) * scale
    pts.push((r * calc.cos(t) * 1cm, r * calc.sin(t) * 1cm))
  }
  path(stroke: stroke-col + w, ..pts)
}

#let swatch(c) = box(baseline: 1.5pt, rect(width: 9pt, height: 9pt,
  radius: 1pt, fill: rgb(c), stroke: 0.3pt + dim))

#let kicker(s) = text(font: "DIN Alternate", size: 8pt,
  tracking: 2pt, fill: cyan, upper(s))

// ---- COVER ----
#page(fill: ink, margin: 0pt, {
  place(center + horizon, dx: 0pt, dy: 0pt, spiral(turns: 4.2,
    scale: 7.0, stroke-col: cyan.lighten(10%), w: 0.6pt))
  place(center + horizon, dx: 0pt, dy: 0pt, spiral(turns: 3.4,
    scale: 4.4, stroke-col: oxide, w: 0.7pt))
  place(center + horizon, dy: -5.2cm,
    text(fill: paper, font: "DIN Alternate", size: 11pt,
      tracking: 6pt)[A VIVERE ASTRA DOSSIER])
  place(center + horizon, dy: -1.3cm,
    text(fill: paper, weight: 700, size: 108pt, tracking: 2pt)[RIG])
  place(center + horizon, dy: 2.6cm, box(width: 13cm,
    text(fill: paper.darken(8%), size: 13pt, style: "italic")[
      #align(center)[The game played in the calm \
      at the heart of a spinning world.]]))
  place(center + bottom, dy: -2.2cm,
    text(fill: dim, font: "DIN Alternate", size: 8.5pt,
      tracking: 1.5pt)[
      #align(center)[FOR THE GM'S CONSIDERATION · PROPOSED CANON \
      THE SPINDLE LEAGUE · NO CLOCK]])
})

// ---- heading styles ----
#set heading(numbering: none)
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  block(above: 0pt, below: 1.1em, {
    kicker("Spindle Dossier")
    v(-0.3em)
    text(size: 30pt, weight: 700, fill: ink, it.body)
    v(-0.2em)
    line(length: 100%, stroke: 1.2pt + oxide)
  })
}
#show heading.where(level: 2): it => block(above: 1.5em, below: 0.7em,
  text(size: 15pt, weight: 700, fill: cyan.darken(15%), it.body))
#show heading.where(level: 3): it => block(above: 1.1em, below: 0.5em,
  text(size: 11.5pt, weight: 700, fill: ink,
    font: "DIN Alternate", it.body))

#set page(numbering: "1", footer: context {
  set text(size: 8pt, fill: dim, font: "DIN Alternate")
  grid(columns: (1fr, 1fr),
    align(left)[RIG · A VIVERE ASTRA DOSSIER],
    align(right)[#counter(page).display() · NO CLOCK])
})

#let pull(body) = block(width: 100%, inset: (x: 1em, y: 0.8em),
  fill: ink, radius: 3pt,
  text(fill: paper, style: "italic", size: 11pt, body))

#let card(name, sub, c1, c2, rec, seed, style, soul) = block(
  width: 100%, breakable: false, inset: 9pt, radius: 3pt,
  stroke: 0.6pt + dim, fill: white.transparentize(40%), {
  grid(columns: (1fr, auto),
    text(weight: 700, size: 11pt, name),
    box(text(font: "DIN Alternate", size: 8pt,
      fill: oxide, weight: 700, seed)))
  text(size: 8.5pt, fill: dim, style: "italic", sub)
  v(3pt)
  grid(columns: (auto, 1fr), column-gutter: 6pt,
    [#swatch(c1)#h(3pt)#swatch(c2)],
    text(size: 8.5pt, font: "DIN Alternate",
      [*#rec* · #style]))
  v(2pt)
  text(size: 9pt, soul)
})

// =====================================================================
#align(center + horizon)[
  #spiral(turns: 2.6, scale: 3.0, stroke-col: dim, w: 0.5pt)
  #v(1.2em)
  #text(size: 13pt, style: "italic", fill: dim)[
    "It was just a thing tired people did on a long shift. \
    Then the chime didn't stop." ]
  #v(0.5em)
  #text(font: "DIN Alternate", size: 8pt, fill: dim,
    tracking: 2pt)[— attributed, origin disputed, see §6]
]

= The Brief

This dossier proposes *rig* — a sport — as Vivere Astra canon. It is not
set dressing. Rig is the argument that a scattered, frightened,
post-collapse humanity, spread across stars it cannot see in real time,
still does *one thing together on purpose, with no clock, for no reason
except that the chime is still ringing.*

The case rests on three claims, each developed in full inside:

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 10pt, row-gutter: 6pt,
  pull[*The venue is canon.* O'Neill earmarked the weightless axis of
    the cylinder for recreation on the first napkin. We occupy a room
    the setting already built.],
  pull[*The physics is the game.* The gravity gradient and the Coriolis
    asymmetry aren't flavor — they generate every rule, every position,
    every team's soul.],
  pull[*The economics is the setting's.* Clockless, audio-first,
    Lira-denominated, jump-bottlenecked, governed by the one faction
    with no army. Rig's money *is* Vivere Astra's money.])

#v(0.5em)
A reader who finishes this can: follow a broadcast, explain why a *rise*
is worth more than a *fall*, name the three GM hooks loaded into the
economy, and seed a 32-team interstellar playoff. The companion artifact
— a playable match in the calm — ships alongside.

#pull[*The pitch in one line.* Rig is what Vivere Astra's humans have
instead of a shared sky.]

= The Physics — this *is* the sport

Sources swept and depth-read: the _npj Microgravity_ trajectory paper
(roulette-curve ball flight), Oikofuge's Coriolis guides, the canonical
O'Neill cylinder, and Jon Bois' _17776_ for tone. The full synthesis
lives in the repository; here is what the physics *decided*.

== The room the universe already built

A regulation cylinder is #sym.tilde 6.4–8 km across, 32 km long, spun
#sym.tilde 28 times an hour for one gravity at the rim — and the long
axis down its center stays *weightless*. Riggers call that volume *the
calm*: a windless, weightless tube tens of kilometers long, a few
hundred meters across, with the far land curving up overhead and nothing
falling at the center.

== The three facts that are the whole game

/ The slope you spend: Effective weight rises smoothly from zero at the
  axis to full at the rim. Distance from the axis is a *resource*. Drop
  rimward — *deep* — for power, speed, commitment; the slope gives no
  refunds. Float axisward — *high* — for time, options, reversibility,
  and softness. Coaches say it in three words: *deep is faithful, high
  is free.*

/ The cruel hand: A spinning world deflects everything that flies.
  Thrown *with* the spin (*fair*), the bell arcs about how your gut
  expects. Thrown *against* it (*cross*), the bell *rises, curves hard,
  and can sail clear around a bend* before it comes down. The whole
  sport is lopsided around this — one goal is honest and cheap, the
  other is against the grain and dear.

/ You cannot fly: In the calm there is no swimming or cutting. You
  change vector exactly two ways: push off something solid, or throw a
  line and haul. Every rigger carries a powered tether-and-claw — a
  grapple, descended from the construction lines the game was born on.
  Rig is a grappling-hook sport at the bones. The hook is the only
  reason humans can play at all.

#pull[Ball flight in the calm follows *roulette curves* — involutes and
Archimedean spirals (the motif on every page of this dossier is not
decoration; it is a thrown bell). The closed-arc *loop* — a throw that
curves back to the hand untouched — is real, and it is the founding
trick and the seven-point score.]

= The Sport

Rig: two crews in the calm; the formal league name is the *Spindle
League*; a match is *a rig*; you *rig up*; a player is *a rigger*.

== The field, the bell, the crew

A regulation field is *640 m* of the calm between two *goal rings*,
sleeved by a soft mesh *skin*; touch it and you are *skinned* (out).
Down the axis run fixed *spars* (clip-points); three notional *gates* —
*first, deep, mouth* — divide it. The two ends are unequal by rule: the
*Faith* end is attacked *with* the spin (honest, heavier, easier); the
*Free* end *against* it (rarer, dearer). *Choosing ends is the deepest
strategic decision in the sport.*

The *bell* is a dense soft sphere that *rings* when it spins true and
*clatters* when it tumbles — read by ear, which is why rig broadcasts on
audio across light-years. Six a side, free substitution at any dead
bell. Positions are *radius jobs*, not field positions:

#table(columns: (auto, auto, 1fr), inset: 6pt, stroke: 0.4pt + dim,
  align: (x, y) => if y == 0 { center } else { left },
  table.header(
    text(font: "DIN Alternate", weight: 700, size: 8.5pt)[POS],
    text(font: "DIN Alternate", weight: 700, size: 8.5pt)[NAME],
    text(font: "DIN Alternate", weight: 700, size: 8.5pt)[JOB]),
  [1], [*Anchor*], [Deep & heavy. Power and last defense. Never floats.],
  [2], [*Spinners* ×2], [Midfield engine. Trade altitude for tempo; run
    the lines; set loop plays. The league's best.],
  [2], [*Wings* ×2], [A *Faithwing* (power, fair side) and a *Freewing*
    (the artist; cross side — the franchise's signature name).],
  [1], [*Reach*], [Goal-ring keeper. May clip and pivot on the ring.
    Sport's lunatics.])

== How play runs — casts, counts, innings

A possession is a *cast* (football's downs, in 3D, with radius as a free
extra axis): three throws to clear the next gate, or turn it over.
Football moves the bell; *baseball decides it.* The instant a defender
contests a bell in flight, play freezes into a one-on-one — *a one* — on
*the count* of three exchanges: complete, or *snatch* or *clatter* for a
turnover *at that radius and direction*. A one fought deep is fast,
violent, final; fought high it is slow, reversible, chess. The slope is
*inside* the duel. A rig box score is a wall of one-on-one lines — the
stat-haunted baseball soul, in flight.

A match is *nine innings*. Tie → *spine*: sudden-death casts at the Free
end, against the world's hand, forever. There is no clock. There has
never been a clock. Three commissioners died proposing one; the league
counts this as a rule.

== Scoring

#table(columns: (auto, auto, 1fr), inset: 7pt, stroke: 0.4pt + dim,
  table.header(
   text(font:"DIN Alternate", weight:700, size:8.5pt)[SCORE],
   text(font:"DIN Alternate", weight:700, size:8.5pt)[WORTH],
   text(font:"DIN Alternate", weight:700, size:8.5pt)[WHAT IT IS]),
  [*Fall*], [#text(fill: cyan, weight: 700)[2]],
    [Bell through the *Faith* (fair) ring. The honest score. The bread.],
  [*Rise*], [#text(fill: oxide, weight: 700)[5]],
    [Through the *Free* (cross) ring, into the world's wrong-handed
     curve. Most riggers never score one in a season.],
  [*Loop*], [#text(fill: oxide, weight: 700)[7]],
    [Score with a closed Coriolis arc — bell untouched in flight, bent
     home on the world's own math. Ends the inning. The home run, the
     no-hitter, and the Hail Mary fused. The crowd goes *silent* to hear
     if the bell still rings.],
  [*Ground*], [#text(fill: dim, weight: 700)[1]],
    [(to the defense) Force an attacker into the skin. The grinding soul
     of great defensive habitats.])

= The Rules in Brief & The Glossary

Enough to follow a match — not the Reg (ten thousand pages, three dead
commissioners). The six ideas a newcomer *must* be told, because they
have no Earth equivalent:

#grid(columns: (1fr, 1fr), column-gutter: 12pt, row-gutter: 7pt,
  pull[*The calm* — the field is a *volume*, not a plane: the weightless
    axis of a spun world.],
  pull[*The slope* — gravity rises rim-ward; your radius is a resource
    you continuously spend.],
  pull[*The asymmetry* — the two goals are physically different. Honest
    cheap *Fall* vs. against-the-grain dear *Rise*.],
  pull[*The Coriolis loop* — thrown things bend, and a *cross* throw can
    fly a closed arc home. The 7.],
  pull[*The line* — no flight in zero-g; the grapple is the *only*
    mobility. Rig is a hook sport.],
  pull[*Reading by ear* — the bell rings true, clatters tumbled; hearing
    is the deciding sense, hence audio-cast.])

== Fouls, short list

*Garrote* — a live line across a flight path; the cardinal sin,
ejection. *Cording* — wrapping a line on a body/line; turnover+advance.
*Skinning* — driving a player into the skin; turnover, repeat ejects.
*Crowding the Reach* — contacting a clipped keeper; free throw from the
mouth. *Slow bell* — stalling to run a clock that *does not exist*; a
foul, because pretending there is a clock is heresy.

== The simplification pass (the user's note, honored)

A designed term and a *shouted* term are different objects. Left is what
the Reg calls it; right is the blunt monosyllable a rigger yells
mid-flight across a bad audio gap. The bluntness is itself canon — the
riggers' fingerprint on the language.

#table(columns: (1fr, auto, 1.3fr), inset: 5.5pt, stroke: 0.4pt + dim,
  table.header(
   text(font:"DIN Alternate",weight:700,size:8pt)[ON PAPER],
   text(font:"DIN Alternate",weight:700,size:8pt)[SHOUTED],
   text(font:"DIN Alternate",weight:700,size:8pt)[NOTE]),
  [antispinward / against the spin], [*cross*], [the glamour word],
  [spinward / with the spin], [*fair*], [the honest side],
  [rimward / toward the skin], [*deep*], ["take it deep"],
  [axisward / toward the axis], [*high*], ["high and free"],
  [the Faithful goal / Free goal], [*Faith / Free*], ["they're a Free team"],
  [a contest / one-on-one], [*a one*], ["won the one"],
  [make the bell tumble], [*clatter*], [defense's oldest cry],
  [take it clean], [*snatch*], ["snatched at the mouth"],
  [Faithwing / Freewing], [*the Faith / the Free*], [a star Free is "their Free"],
  [grounded into the skin], [*skinned*], ["he got skinned"],
  [extra spine / overtime], [*spine*], ["going to spine"],
  [the Spindle Regulation], [*the Reg*], [rig's metric system],
  [a regulation match], [*a rig*], ["good rig tonight"])

= The Economics

Rig's money is strange in exactly the way the setting is strange — and
every line is a GM hook.

== The revenue stack

+ *Call rights — the biggest line.* A live system-to-system feed is
  physically impossible (lightspeed). Rig sells *the call, not the
  game*, and the *re-call* — a system's own famous voice re-performing
  a jumped-in match from the chime log days later — is the single
  highest-rated media product in human space. Rig's broadcast is a
  *cover version*, and that is normal.
+ *The gate, by radius band.* Seats priced along the gradient: heavy,
  cheap, loud rim-band (the soul) up to weightless axis boxes (the
  money, where deals get done). The stadium's price gradient *is* the
  field's gradient.
+ *Certification & the Reg.* The Concordat-kept body charges habitats to
  certify regulation spindles and audits the reference value. Small
  money, enormous power.
+ *The in-play market, in Lira.* No clock means no garbage time and no
  kneeling the bell — every cast is live value. Rig is the substrate of
  the largest legal in-play wagering market in the settled systems,
  denominated in *Lira* because the Lira already runs in the seams
  between everyone's flags. Rig betting is a real fraction of why the
  Concordat's currency stays liquid where its flag never flies.

== The costs — where the campaign hooks live

#grid(columns: (1fr, 1fr, 1fr), column-gutter: 9pt, row-gutter: 6pt,
  pull[*Travel is the budget.* A road trip is an O-drive jump: exotic
    matter, weeks down, real Jump cost. A schedule is a colonial supply
    line. A road trip can be *a lead.*],
  pull[*The spindle is a soft target.* A sabotaged regulation spindle is
    a dark stadium and a humiliated faction with no publicized death.
    NISO treats it as infrastructure security.],
  pull[*Riggers are dual-use talent.* A great Freewing thinks in
    Coriolis the way aggies thought in everything. Rig academy →
    astronautics corps → (for a quiet few) France, or Ananke.])

#pull[The shape, in one line: rig is *clockless, audio-first,
Lira-denominated, jump-bottlenecked,* governed by the faction with no
army — Vivere Astra's economics with a bell in them. That is the
argument for canon.]

= The Spindle League — 32 Franchises

Style is not chosen; it is *spun in*. Big slow cylinders breed patient
*Fall* power; small fast stations breed *Rise* chaos; mid cylinders
breed the *ground* line broker game. Faction key: *FRM* Frontier
Republic of Michigan · *CNE* Commonwealth of New England · *CON* the
Concordat · *BLT* Belt independents · *EUR* Old-Europe diaspora · *REF*
the governing body.

== Sol Conference — "First Spin"

#grid(columns: (1fr, 1fr), column-gutter: 9pt, row-gutter: 9pt,
 card("Tuebor Spin Detroiters","Tuebor — great slow Frontier ag cylinder","#3f5a3a","#d4602a","27–6","#1","FRM · Fall dynasty")[The sport's dynasty. Granite Anchor, the honest 2 ground out a hundred times. Neutrals are bored; Tuebor spins on.],
 card("Boston Commons Reach","CNE flagship — balanced","#1f3a93","#f4f1ea","24–9","#2","CNE · Line")[Tuebor vs. Commons *is* the great rivalry — Fall dynasty vs. line craft, Frontier vs. Commonwealth.],
 card("Roma Aeterna Brokers","Concordat Sol flagship — balanced","#5b2a86","#c9a24a","23–10","#3","CON · Ground/away")[The infuriating broker rig. Wins road trophies nobody enjoys watching. The Lira travels; so do its fans.],
 card("Manchester Memorial","CNE — named for the lost colony ship","#6d5a9c","#8a8f99","22–11","#5","CNE · Grief")[Exists as remembrance. Every match opens on a silent bell. When they run, even Frontier crowds stand.],
 card("Cascade Works Foremen","FRM Sol heavy-industry cylinder","#4a4f57","#e4c14a","21–12","#11","FRM · Fall grind")[Blue-collar, clatter-heavy. They make your bell ring ugly. Loudest rim-band in the league.],
 card("Lira Free Company","Concordat — no fixed cylinder","#7a7f88","#c9a24a","20–13","#14","CON · Adaptive")[No home, by design — rents calm wherever the Lira runs. Belt operators fly it as a flag of convenience.],
 card("Saginaw Bell","FRM mid-old Sol cylinder","#e9e0c8","#7a1f2b","18–15","—","FRM · Fall tempo")[The connoisseur's team. Cleanest passing in Sol; scores least dramatically.],
 card("Veracity Yard Probes","FRM Sol shipyard cylinder","#0c0f1a","#1aa6b7","16–17","—","FRM · Loop eng.")[Built the Veracity probe; does Coriolis math for fun. Lost a play-in on a clattered bell. Gallows humor.])

#grid(columns: (1fr, 1fr), column-gutter: 9pt, row-gutter: 9pt,
 card("Plymouth Counter","CNE Sol mid cylinder","#3a4049","#d8a93a","20–13","#—","CON-adj · Line")[Old Commonwealth money, very smug, deservedly. Brutal Reach play.],
 card("New Haven Ear","CNE — officiating academy","#2f5d3a","#f0ece0","19–14","#—","CNE · Legalist")[They train the Ears, then win the Ears' calls. Everyone hates this. It is canon.],
 card("Vatican Spindle","Concordat ceremonial home club","#f4f1ea","#c9a24a","17–16","—","REF · Ceremony")[Institution more than contender. The joke writes itself; the joke is load-bearing, like the Lira.],
 card("Trieste Counterweight","Concordat secondary Sol cylinder","#1f6f78","#9c4a2a","15–18","—","CON · Defensive")[The Concordat's farm of brokers. Dull. Effective.])

== The Near Reach — "First Light"

#grid(columns: (1fr, 1fr), column-gutter: 9pt, row-gutter: 9pt,
 card("Toliman Gradient","Alpha Cen B — small fast station","#f0f0f0","#c0392b","25–8","#8","FRM · Rise chaos")[Proved the colonies could beat Sol. Its Freewing is the most famous rigger alive. Feral, sleepless, evangelical.],
 card("Tau Ceti Mare Nostrum","Concordat flagship colony — balanced","#1f6fa8","#c9a24a","24–9","#4","CON · Broker")[The Concordat's *real* contender. Plays for the away point and the draw and wins the Jump doing it.],
 card("Luhman Orbital Riot","FRM Luhman 16 — born from the riot","#c0392b","#f0ece0","23–10","#6","FRM · Rise/pol")[A labor movement that learned to score. The most dangerous low seed. France-watchers watch this fanbase.],
 card("Hebat Resolve","CNE Fomalhaut — the bombed colony","#eceae0","#4a4f57","22–11","#9","CNE · Defiant")[Sister-grief to Manchester. Their meetings are the most-listened rig in human space. Ananke eyes the spindle.],
 card("Centauri b Hammerline","FRM Alpha Cen colony — mid-fast","#a8731f","#15171c","21–12","#10","FRM · Rise power")[Out-produces the Concordat; plays like it knows. Wins 7s by force.],
 card("Wolf 359 Migration","CNE Wolf 359 — near Tartarus","#5b3a86","#d8c24a","20–13","#15","CNE · Rise flock")[Whole crews move like the macrofauna in Tartarus's clouds. The most coordinated high game in the league.],
 card("Fomalhaut Dagon Wake","CNE Fomalhaut — in sight of dead Dagon","#3a3f47","#7a1f1f","19–14","#—","CNE · Haunted")[Plays under what an RKV did to a living world. NISO dislikes keeping Dagon in front of 40,000 a night.],
 card("Gliese 65 Twinstar","CNE binary-system colony","#b8801f","#1f3a5a","18–15","—","CNE · Line")[Balanced line. Two suns, two interlocked rings.])

#grid(columns: (1fr, 1fr), column-gutter: 9pt, row-gutter: 9pt,
 card("Proxima Long Shift","FRM Proxima outpost — brutal station","#7a1f1f","#15171c","17–16","—","FRM · Rise")[Plays like the shift that invented the game never ended. Frontier romance incarnate.],
 card("Lacaille Longjump","CNE deeper colony","#0c0f1a","#d4602a","16–17","—","CNE · Travel")[Always the away team; ground rig, travel-hardened.],
 card("82 Eridani Secondi","Concordat secondary colony","#9c5a2a","#c9a24a","15–18","—","CON · Junior")[The Concordat's far farm of brokers.],
 card("Frontier Veracity Deep","FRM deep colony — Centauri edge","#5a5f57","#3f5a3a","17–16","—","FRM · Fall")[Isolated, stubborn, far. The edge of the push.])

== The Far Reach & the Belt — "Deep Jump"

#grid(columns: (1fr, 1fr), column-gutter: 9pt, row-gutter: 9pt,
 card("Kuiper Long Call","Deepest, darkest Belt franchise","#0c0d11","#f0ece0","22–11","#7","BLT · Audio ground")[Out where the broadcast *is* the team. Their re-call is the highest-rated media in human space. Richest "small" club.],
 card("Belt Free Local 9","Belt independent, syndicate-run","#e07a1f","#15171c","21–12","#12","BLT · Rise lunatic")[The purest chaos rig. Half the league's best came up here — *and some people the league won't name.*],
 card("Ceres Counterspin","Belt's largest hub station","#7a7f88","#9cd0e0","20–13","#13","BLT · Contrarian")[Attacks the *Free* end by preference — the most expensive way to play — because Belters are like that.],
 card("Vesta Tool-Bag Originals","Belt — claims oldest folk-rig lineage","#9c6a1f","#15171c","18–15","#16","BLT · Ceremonial")["WE THREW THE FIRST LOOP." Every origin-myth argument routes through Vesta's marketing department.],
 card("Eros Garrote","Belt — the beloved villains","#6e1f1f","#d8c24a","19–14","#—","BLT · Dirty")[Foul-prone, magnetic, the team you boo and can't stop hearing. Best road-call ratings in the league.],
 card("Silesia Exiles","Old-Europe diaspora (ASHFIELD descendants)","#3a3f47","#9c8a6a","19–14","#—","EUR · Mournful")[Old Europe's only seat at the table. The neutral's favorite. A bridge between Kanzo's campaigns.],
 card("Luhman 16 Brownline","FRM Luhman brown-dwarf system","#5a2a2a","#c98a3a","17–16","—","BLT-adj · Rise")[Low-light audio legends — broadcasts better than it looks.],
 card("The Reference Eleven","Governing body's neutral all-star side","#f4f1ea","#8a8f99","exh.","#—","REF · Platonic")[Plays only the reference spin nobody's home uses. No fans by birth, only by belief. The Reg in a jersey.])

= The Jump — Current Playoff Seeding

We are on the eve of the Jump: the top 16 cross into interstellar
single-elimination. Higher seed hosts on its certified spindle; *the
lower seed pays the jump* (exotic matter + weeks of transit) — travel is
the bracket's hidden bracket, per canon.

#table(columns: (auto, 1fr, auto, auto, 1.1fr), inset: 5.5pt,
  stroke: 0.4pt + dim,
  align: (x,y) => if x == 0 or x == 2 or x == 3 { center } else { left },
  table.header(..("SEED","TEAM","FAC","REC","STYLE").map(s =>
   text(font:"DIN Alternate", weight:700, size:8pt, s))),
  [1],[Tuebor Spin Detroiters],[FRM],[27–6],[Fall dynasty],
  [2],[Boston Commons Reach],[CNE],[24–9],[Line],
  [3],[Roma Aeterna Brokers],[CON],[23–10],[Ground/away],
  [4],[Tau Ceti Mare Nostrum],[CON],[24–9],[Broker],
  [5],[Manchester Memorial],[CNE],[22–11],[Ground (grief)],
  [6],[Luhman Orbital Riot],[FRM],[23–10],[Rise (political)],
  [7],[Kuiper Long Call],[BLT],[22–11],[Audio ground],
  [8],[Toliman Gradient],[FRM],[25–8],[Rise chaos],
  [9],[Hebat Resolve],[CNE],[22–11],[Ground (defiant)],
  [10],[Centauri b Hammerline],[FRM],[21–12],[Rise power],
  [11],[Cascade Works Foremen],[FRM],[21–12],[Fall grind],
  [12],[Belt Free Local 9],[BLT],[21–12],[Rise lunatic],
  [13],[Ceres Counterspin],[BLT],[20–13],[Rise contrarian],
  [14],[Lira Free Company],[CON],[20–13],[Adaptive ground],
  [15],[Wolf 359 Migration],[CNE],[20–13],[Rise flock],
  [16],[Vesta Tool-Bag Originals],[BLT],[18–15],[Ceremonial Rise])

#pull[*Seeding note.* Toliman is 25–8 yet seeded #8: the Jump weights
*strength of jump* — wins over teams that had to cross interstellar
distance to reach you count less than wins earned on the road. The
Concordat designed the tiebreak. Of course they did.]

== First-round storylines the calls will sell

#grid(columns: (1fr, 1fr), column-gutter: 10pt, row-gutter: 6pt,
 pull[*1 Tuebor v 16 Vesta* — the dynasty vs. the origin myth. "We
   defend" vs. "we threw the first loop." Whoever loses, marketing wins.],
 pull[*5 Manchester v 12 Belt Free Local 9* — the grief franchise vs.
   the dual-use talent fountain. NISO has people in both rim-bands. A
   lead waiting to happen.],
 pull[*6 Luhman Orbital Riot v 11 Cascade Works* — the riot vs. the
   foremen, on national audio, France-backed-riot optics. Kanzo can
   light this fuse anytime.],
 pull[*8 Toliman v 9 Hebat* — chaos vs. grief, the two best low seeds,
   the connoisseur's pick for rig of the year.])

= Off the Board — hooks the league won't print

#grid(columns: (1fr, 1fr), column-gutter: 10pt, row-gutter: 7pt,
 pull[*The Directorate Circuit.* France's breakaway pariah league —
   O-drive-funded, an aesthetic of *forced Rises* the Spindle League
   calls decadent. Runs its own spin, rejects the reference. Recognizing
   the Circuit is a diplomatic act; a rigger who jumps to it has
   *defected.*],
 pull[*The Ananke question.* Ananke has tried for a championship spindle
   once. The league does not discuss it. NISO does not either. The
   reader is told only this much — on purpose (17776's witnessing
   restraint, as a plot device).],
 pull[*The talent pipeline.* Rig academy → astronautics corps → for a
   quiet few, France or Ananke. The best Freewings think in Coriolis the
   way aggies used to think in everything. The league's unspoken anxiety
   is the campaign's, with a jersey on.],
 pull[*Certification as recognition.* A new colony's first certified
   spindle says it *belongs.* A spindle lost to sabotage or riot is a
   bloodless humiliation on interstellar audio. Every certification is a
   small treaty. Hand to the GM.])

= Bibliography & Colophon

#text(size: 9pt)[
*Physics & venue.* _Simplified equations for object trajectories in
rotating space habitats_, npj Microgravity (PMC10570307). Oikofuge,
_Coriolis Effect in a Rotating Space Habitat_ (+ axial supplement).
Wikipedia, _O'Neill cylinder_; NSS O'Neill settlement page.
*Tone & structure.* Jon Bois, _17776 / What Football Will Look Like in
the Future_, SB Nation. *Founding template.* Histories of basketball
(Naismith), association-football codification, pickup/folk games;
Space Games Federation & ISS improvised sport. *Setting.* Vivere Astra
canon (Kanzokax, GM) — `vivere_astra.md`, `faction.md`, `strategy.md`,
`campaign_log.md`.

#v(1em)
#line(length: 100%, stroke: 0.4pt + dim)
#v(0.6em)
#grid(columns: (1fr, auto),
  text(size: 8pt, fill: dim, font: "DIN Alternate")[
    Set in New Computer Modern. Spirals are computed Archimedean
    roulettes — every one is a thrown bell. Built with Typst.],
  text(size: 8pt, fill: oxide, font: "DIN Alternate",
    weight: 700)[NO CLOCK])
]

#align(center + horizon)[
  #v(2em)
  #spiral(turns: 3.0, scale: 2.4, stroke-col: oxide, w: 0.7pt)
  #v(1em)
  #text(style: "italic", fill: dim, size: 11pt)[
    The chime is still ringing.]
]
