import { COLORS } from '../constants'

const VARIANTS = {
  primary: {
    bg: `linear-gradient(180deg, ${COLORS.accentHover} 0%, ${COLORS.accent} 100%)`,
    hoverBg: `linear-gradient(180deg, ${COLORS.accentHover} 0%, ${COLORS.accentHover} 100%)`,
    color: COLORS.bg,
    border: 'transparent',
    shadow: `0 1px 2px rgba(0,0,0,0.3), 0 4px 14px ${COLORS.glowAccent}`,
    hoverShadow: `0 1px 2px rgba(0,0,0,0.3), 0 6px 20px ${COLORS.glowAccentStrong}`,
  },
  secondary: {
    bg: COLORS.surface2, hoverBg: COLORS.hover, color: COLORS.textSecondary, border: COLORS.borderDefault,
    shadow: 'none', hoverShadow: 'none',
  },
  danger: {
    bg: 'transparent', hoverBg: COLORS.dangerBg, color: COLORS.danger, border: COLORS.dangerBorder,
    shadow: 'none', hoverShadow: 'none',
  },
}

// Single reusable button primitive so every CTA in the app shares one
// visual system instead of each component re-inventing padding/radius/
// hover behavior inline (spec: "avoid massive monolithic components,
// create/reuse design primitives").
export default function Button({ variant = 'secondary', className = '', style = {}, children, ...props }) {
  const v = VARIANTS[variant] ?? VARIANTS.secondary
  return (
    <button
      {...props}
      className={`rounded-[8px] border px-4 py-2 text-[13px] font-medium transition-all duration-150 ease-out disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:translate-y-0 ${className}`}
      style={{ background: v.bg, color: v.color, borderColor: v.border, boxShadow: v.shadow, ...style }}
      onMouseEnter={(e) => {
        if (props.disabled) return
        e.currentTarget.style.background = v.hoverBg
        e.currentTarget.style.boxShadow = v.hoverShadow
        e.currentTarget.style.transform = 'translateY(-1px)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = v.bg
        e.currentTarget.style.boxShadow = v.shadow
        e.currentTarget.style.transform = 'translateY(0)'
      }}
    >
      {children}
    </button>
  )
}
