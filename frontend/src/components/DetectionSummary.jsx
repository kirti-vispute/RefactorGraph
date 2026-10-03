import { ArrowRightLeft, Boxes, CheckCircle2, Ruler } from 'lucide-react'
import { COLORS, SMELL_LABELS } from '../constants'

const CARD_ACCENT = {
  long_method: COLORS.warning,
  feature_envy: COLORS.danger,
  god_class: '#9B7CFF',
}

const CARD_ICON = {
  long_method: Ruler,
  feature_envy: ArrowRightLeft,
  god_class: Boxes,
}

// Only predicted===true entries count as a "detection" (section 8 of the
// spec) -- the full per-node probability list still exists in `results`
// for the graph, but counts/rows here are real detections only.
function detectionsFor(results, task) {
  return results[task]
    .filter((p) => p.predicted)
    .slice()
    .sort((a, b) => b.probability - a.probability)
}

export default function DetectionSummary({ results, selected, onSelect }) {
  const tasks = Object.keys(SMELL_LABELS)

  return (
    <section>
      <h2 className="mb-3 text-[15px] font-semibold" style={{ color: COLORS.textPrimary }}>Detection Summary</h2>

      <div className="grid grid-cols-3 gap-3">
        {tasks.map((task, i) => {
          const count = detectionsFor(results, task).length
          const Icon = CARD_ICON[task]
          return (
            <div
              key={task}
              className="relative animate-[fadeInUp_0.4s_ease-out_both] overflow-hidden rounded-[10px] border px-4 py-3 text-center transition-transform duration-150 hover:-translate-y-0.5"
              style={{
                backgroundColor: COLORS.surface,
                borderColor: count > 0 ? `${CARD_ACCENT[task]}55` : COLORS.borderDefault,
                boxShadow: count > 0 ? `0 0 0 1px ${CARD_ACCENT[task]}22, 0 4px 16px ${CARD_ACCENT[task]}1a` : 'none',
                animationDelay: `${i * 40}ms`,
              }}
            >
              <Icon size={14} className="mx-auto mb-1 opacity-70" style={{ color: count > 0 ? CARD_ACCENT[task] : COLORS.textDisabled }} />
              <div className="text-2xl font-semibold tabular-nums" style={{ color: count > 0 ? CARD_ACCENT[task] : COLORS.textDisabled }}>
                {count}
              </div>
              <div className="mt-1 text-[11px] font-medium uppercase tracking-wide" style={{ color: COLORS.textMuted }}>
                {SMELL_LABELS[task]}
              </div>
            </div>
          )
        })}
      </div>

      <ul className="mt-3 divide-y overflow-hidden rounded-[10px] border" style={{ borderColor: COLORS.borderDefault, backgroundColor: COLORS.surface }}>
        {tasks.flatMap((task) => detectionsFor(results, task)).length === 0 && (
          <li className="flex items-center gap-2 px-4 py-4 text-[13px]" style={{ color: COLORS.success }}>
            <CheckCircle2 size={15} />
            No code smells detected.
          </li>
        )}
        {tasks.map((task) =>
          detectionsFor(results, task).map((p) => {
            const isSelected = selected?.node_id === p.node_id && selected?.task === task
            return (
              <li key={`${task}:${p.node_id}`} style={{ borderColor: COLORS.divider }}>
                <button
                  type="button"
                  onClick={() => onSelect(task, p)}
                  className="flex w-full items-center justify-between gap-3 border-l-2 px-4 py-2.5 text-left text-[13px] transition-colors"
                  style={{
                    backgroundColor: isSelected ? COLORS.hover : 'transparent',
                    borderLeftColor: isSelected ? CARD_ACCENT[task] : 'transparent',
                  }}
                  onMouseEnter={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = COLORS.surface2 }}
                  onMouseLeave={(e) => { if (!isSelected) e.currentTarget.style.backgroundColor = 'transparent' }}
                >
                  <span className="flex items-center gap-2 overflow-hidden">
                    <span
                      className="shrink-0 rounded-[4px] px-1.5 py-0.5 text-[10px] font-semibold uppercase"
                      style={{ backgroundColor: COLORS.elevated, color: CARD_ACCENT[task] }}
                    >
                      {SMELL_LABELS[task]}
                    </span>
                    <span className="truncate font-mono" style={{ color: COLORS.textPrimary }}>{p.name}</span>
                  </span>
                  <span className="shrink-0 font-mono tabular-nums text-xs" style={{ color: COLORS.textSecondary }}>
                    {(p.probability * 100).toFixed(1)}%
                  </span>
                </button>
              </li>
            )
          }),
        )}
      </ul>
    </section>
  )
}
