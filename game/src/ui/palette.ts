// Canon palette — every colour decision in RIG flows from here.
// World +X spin axis; (y,z) cross-section; skin radius R=45.
// Hex numbers for Three.js material colours; css strings for DOM/canvas.

export const PAL = {
  // backgrounds
  bg:     0x11131a,
  bgCss:  '#11131a',

  // team/scoring accent pair
  cyan:     0x1aa6b7,
  cyanCss:  '#1aa6b7',

  orange:     0xd4602a,
  orangeCss:  '#d4602a',

  // paper (UI surfaces, readable text)
  paper:     0xf4f1ea,
  paperCss:  '#f4f1ea',

  // dim (secondary lines, ghost elements)
  dim:     0x6b7079,
  dimCss:  '#6b7079',

  // derived / meta
  faithEnd:   0x1aa6b7, // spinward ring = cyan
  freeEnd:    0xd4602a, // antispinward ring = orange
  home:       0x1aa6b7, // home team tint
  away:       0xd4602a, // away team tint

  // P1 highlight (the controlled rigger)
  p1Highlight:    0xf4f1ea,
  p1HighlightCss: '#f4f1ea',

  // reticle states
  reticleValid:   0x1aa6b7,
  reticleInvalid: 0xd4602a,
  reticleOORange: 0x6b7079,
} as const;

/** Role accent colours — a subtle tint on the low-poly figure. */
export const ROLE_TINT: Record<string, number> = {
  anchor:    0x2a3a6b, // deep navy
  spinner:   0x1aa6b7, // cyan midfield
  faithwing: 0x3ad4b7, // teal-green power
  freewing:  0xd4a02a, // gold artist
  reach:     0xa72ab7, // violet lunatic
};
