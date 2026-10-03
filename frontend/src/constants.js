// Fixed, project-defined mappings -- not model output, not fabricated.
// The app recommends a refactoring action; it never rewrites source code.
export const SMELL_LABELS = {
  long_method: 'Long Method',
  feature_envy: 'Feature Envy',
  god_class: 'God Class',
}

export const REFACTORING_RECOMMENDATIONS = {
  long_method: 'Extract Method',
  feature_envy: 'Move Method',
  god_class: 'Split Class',
}

// Only the types graph_builder.py can actually produce ("variable" is
// deliberately excluded from the schema -- see backend docstring).
export const NODE_TYPES = ['module', 'class', 'method', 'function', 'attribute', 'parameter', 'import']

// --- Design system tokens -------------------------------------------------
// Professional dark developer-tool palette (Linear/Vercel/Raycast-adjacent).
export const COLORS = {
  bg: '#0F1219',
  surface: '#171B24',
  surface2: '#1D222C',
  elevated: '#222732',
  hover: '#282E39',

  borderDefault: '#2C323D',
  borderStrong: '#373E4B',
  divider: '#242933',

  textPrimary: '#F4F7FA',
  textSecondary: '#A7AFBC',
  textMuted: '#707987',
  textDisabled: '#4D5562',

  accent: '#7C8CFF',
  accentHover: '#91A0FF',
  accentMuted: '#252A4A',

  success: '#3DDC97',
  warning: '#F2C94C',
  danger: '#FF6B6B',
  info: '#5BA7FF',

  // Same +15-ish lightness shift applied to the danger/error surfaces as
  // the base bg/surface scale above, so error banners don't read as a
  // leftover darker patch against the now-lighter theme.
  dangerBg: '#31191D',
  dangerBorder: '#532A35',

  // Soft glows/shadows layered on top of the flat palette above -- additive
  // only, nothing existing was renamed or removed (several tests assert
  // exact COLORS.* values elsewhere, e.g. CodeGraph.test.jsx).
  glowAccent: 'rgba(124, 140, 255, 0.18)',
  glowAccentStrong: 'rgba(124, 140, 255, 0.35)',
  shadowCard: '0 1px 2px rgba(0, 0, 0, 0.4), 0 8px 24px rgba(0, 0, 0, 0.28)',
  shadowCardHover: '0 1px 2px rgba(0, 0, 0, 0.4), 0 12px 32px rgba(0, 0, 0, 0.4)',
}

export const NODE_TYPE_COLORS = {
  module: '#5B8DEF',
  class: '#9B7CFF',
  method: '#6F9CFF',
  function: '#45C2B8',
  attribute: '#D9A441',
  parameter: '#62B88F',
  import: '#F27DAF',
}

export const NODE_TYPE_SHAPES = {
  module: 'round-rectangle',
  class: 'round-rectangle',
  method: 'round-rectangle',
  function: 'round-rectangle',
  attribute: 'ellipse',
  parameter: 'ellipse',
  import: 'diamond',
}

export const EDGE_TYPE_COLORS = {
  contains: '#566174',
  calls: '#6F9CFF',
  uses: '#45C2B8',
  accesses: '#D9A441',
  inherits: '#9B7CFF',
  imports: '#707987',
  belongs_to: '#566174',
}

export const NOT_AVAILABLE = 'Not available from analysis.'
