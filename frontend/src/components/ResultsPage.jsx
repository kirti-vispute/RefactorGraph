import { useMemo, useState } from 'react'
import { COLORS } from '../constants'
import AnalysisSummary from './AnalysisSummary'
import Button from './Button'
import CodeGraph from './CodeGraph'
import DetectionSummary from './DetectionSummary'
import SmellDetailPanel from './SmellDetailPanel'

const TASKS = ['long_method', 'feature_envy', 'god_class']

export default function ResultsPage({ analysis, onReset }) {
  const [selection, setSelection] = useState(null)

  const predictionsByNodeId = useMemo(() => {
    const map = new Map()
    for (const task of TASKS) {
      for (const prediction of analysis[task]) {
        const entries = map.get(prediction.node_id) ?? []
        entries.push({ task, prediction })
        map.set(prediction.node_id, entries)
      }
    }
    return map
  }, [analysis])

  function handleSelect(task, prediction) {
    setSelection({ task, prediction })
  }

  return (
    <div className="min-h-screen">
      <header
        className="sticky top-0 z-10 border-b px-6 py-3 backdrop-blur"
        style={{ backgroundColor: `${COLORS.bg}f2`, borderColor: COLORS.borderDefault, boxShadow: '0 4px 24px rgba(0,0,0,0.3)' }}
      >
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <div className="flex items-center gap-2.5">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS.accent, boxShadow: `0 0 10px ${COLORS.glowAccentStrong}` }} />
            <div>
              <div className="text-[11px] font-medium uppercase tracking-[0.08em]" style={{ color: COLORS.textMuted }}>
                RefactorGraph
              </div>
              <h1 className="text-[15px] font-semibold" style={{ color: COLORS.textPrimary }}>Analysis Results</h1>
            </div>
          </div>
          <Button variant="secondary" onClick={onReset} className="!py-1.5">
            Analyze Another File
          </Button>
        </div>
      </header>

      <div className="animate-fade-in mx-auto max-w-6xl space-y-8 px-6 py-8">
        <AnalysisSummary summary={analysis.summary} />
        <DetectionSummary results={analysis} selected={selection} onSelect={handleSelect} />
        <SmellDetailPanel selection={selection} />
        <CodeGraph graph={analysis.graph} predictionsByNodeId={predictionsByNodeId} focusSelection={selection} />
      </div>
    </div>
  )
}
