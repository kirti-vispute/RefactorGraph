import { AlertTriangle, Box, Braces, FunctionSquare, GitBranch, Layers } from 'lucide-react'
import { COLORS } from '../constants'

function Stat({ label, value, accent, icon: Icon, delay }) {
  return (
    <div
      className="group relative animate-[fadeInUp_0.4s_ease-out_both] overflow-hidden rounded-[10px] border px-4 py-3 transition-all duration-150 hover:-translate-y-0.5"
      style={{ backgroundColor: COLORS.surface, borderColor: COLORS.borderDefault, animationDelay: `${delay}ms` }}
      onMouseEnter={(e) => (e.currentTarget.style.boxShadow = COLORS.shadowCard)}
      onMouseLeave={(e) => (e.currentTarget.style.boxShadow = 'none')}
    >
      <div className="absolute inset-x-0 top-0 h-[2px]" style={{ backgroundColor: accent, opacity: 0.7 }} />
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-xl font-semibold tabular-nums" style={{ color: COLORS.textPrimary }}>{value}</div>
          <div className="mt-0.5 text-[11px] font-medium uppercase tracking-wide" style={{ color: COLORS.textMuted }}>{label}</div>
        </div>
        <Icon size={16} className="mt-0.5 shrink-0 opacity-70" style={{ color: accent }} />
      </div>
    </div>
  )
}

// Every value here comes directly from AnalyzeResponse.summary -- nothing
// on this panel is computed or estimated on the frontend.
export default function AnalysisSummary({ summary }) {
  return (
    <section>
      <h2 className="mb-3 text-[15px] font-semibold" style={{ color: COLORS.textPrimary }}>Analysis Summary</h2>
      <div className="mb-3 truncate font-mono text-[13px]" style={{ color: COLORS.textSecondary }}>{summary.filename}</div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Classes" value={summary.n_classes} accent={COLORS.info} icon={Box} delay={0} />
        <Stat label="Methods" value={summary.n_methods} accent={COLORS.info} icon={FunctionSquare} delay={40} />
        <Stat label="Functions" value={summary.n_functions} accent={COLORS.info} icon={Braces} delay={80} />
        <Stat label="Graph Nodes" value={summary.n_nodes} accent={COLORS.textMuted} icon={Layers} delay={120} />
        <Stat label="Graph Edges" value={summary.n_edges} accent={COLORS.textMuted} icon={GitBranch} delay={160} />
        <Stat
          label="Smells Detected"
          value={summary.n_detected}
          accent={summary.n_detected > 0 ? COLORS.danger : COLORS.success}
          icon={AlertTriangle}
          delay={200}
        />
      </div>
    </section>
  )
}
