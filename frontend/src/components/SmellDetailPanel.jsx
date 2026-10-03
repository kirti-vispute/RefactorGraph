import { MousePointerClick } from 'lucide-react'
import { COLORS, NOT_AVAILABLE, REFACTORING_RECOMMENDATIONS, SMELL_LABELS } from '../constants'

const TASK_ACCENT = {
  long_method: COLORS.warning,
  feature_envy: COLORS.danger,
  god_class: '#9B7CFF',
}

function Field({ label, children, valueColor }) {
  return (
    <div>
      <dt className="text-[11px] font-medium uppercase tracking-wide" style={{ color: COLORS.textMuted }}>{label}</dt>
      <dd className="mt-0.5 text-[13px]" style={{ color: valueColor ?? COLORS.textPrimary }}>{children}</dd>
    </div>
  )
}

// Every value rendered here comes straight from the selected prediction
// object the backend returned -- nothing is invented client-side, and
// anything the backend didn't provide falls back to NOT_AVAILABLE.
export default function SmellDetailPanel({ selection }) {
  if (!selection) {
    return (
      <section>
        <h2 className="mb-3 text-[15px] font-semibold" style={{ color: COLORS.textPrimary }}>Detailed Smell Analysis</h2>
        <p
          className="flex flex-col items-center justify-center gap-2 rounded-[10px] border px-4 py-8 text-center text-[13px]"
          style={{ backgroundColor: COLORS.surface, borderColor: COLORS.borderDefault, color: COLORS.textMuted }}
        >
          <MousePointerClick size={18} className="opacity-60" />
          Select a detected smell above to see its details.
        </p>
      </section>
    )
  }

  const { task, prediction: p } = selection
  const methodOrFunction = p.node_type === 'class' ? NOT_AVAILABLE : p.name.split('.').pop()
  const lineRange = p.line_start != null && p.line_end != null ? `${p.line_start}–${p.line_end}` : NOT_AVAILABLE

  const accent = TASK_ACCENT[task] ?? COLORS.accent

  return (
    <section>
      <h2 className="mb-3 text-[15px] font-semibold" style={{ color: COLORS.textPrimary }}>Detailed Smell Analysis</h2>
      <div
        className="relative overflow-hidden rounded-[10px] border p-5"
        style={{ backgroundColor: COLORS.surface, borderColor: COLORS.borderDefault, boxShadow: COLORS.shadowCard }}
      >
        <div className="absolute inset-x-0 top-0 h-[2px]" style={{ backgroundColor: accent }} />
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
          <Field label="Smell" valueColor={accent}>{SMELL_LABELS[task]}</Field>
          <Field label="Model Score">{(p.probability * 100).toFixed(1)}%</Field>
          <Field label="File">
            <span className="font-mono">{p.file}</span>
          </Field>
          <Field label="Class">
            <span className="font-mono">{p.class_name ?? NOT_AVAILABLE}</span>
          </Field>
          <Field label="Method / Function">
            <span className="font-mono">{methodOrFunction}</span>
          </Field>
          <Field label="Source Lines">{lineRange}</Field>
          <div className="sm:col-span-2">
            <Field label="Structural Metrics for This Node">
              <span className="font-mono">{p.explanation || NOT_AVAILABLE}</span>
            </Field>
          </div>
          <div className="sm:col-span-2">
            <Field label="Recommended Refactoring">
              <span
                className="inline-block rounded-[6px] px-2 py-1 text-[13px] font-semibold"
                style={{ backgroundColor: COLORS.accentMuted, color: COLORS.accentHover }}
              >
                {REFACTORING_RECOMMENDATIONS[task]}
              </span>
            </Field>
          </div>
        </dl>
        <p className="mt-4 border-t pt-3 text-xs" style={{ borderColor: COLORS.divider, color: COLORS.textMuted }}>
          Model Score is the model's raw prediction output, not a measured accuracy rate or a
          calibrated probability of correctness — no reliability study has been run to confirm
          that e.g. a 90% score is right 90% of the time. The structural metrics above are real,
          computed directly from this node's own code, but the model's decision also draws on
          semantic (CodeBERT) and graph-relationship signals not summarized as numbers here.
          RefactorGraph recommends a refactoring action based on the detected smell — it does not
          automatically rewrite or refactor your source code.
        </p>
      </div>
    </section>
  )
}
