import { Loader2, Network, Upload } from 'lucide-react'
import { useRef, useState } from 'react'
import { COLORS, SMELL_LABELS } from '../constants'
import Button from './Button'

const SMELL_CHIP_COLOR = {
  long_method: COLORS.warning,
  feature_envy: COLORS.danger,
  god_class: '#9B7CFF',
}

const PLACEHOLDER = `class OrderProcessor:
    def process(self, order):
        ...
`

export default function CodeInputPage({ onAnalyze, loading, apiError }) {
  const [source, setSource] = useState('')
  const [filename, setFilename] = useState('input.py')
  const [localError, setLocalError] = useState(null)
  const fileInputRef = useRef(null)

  function handleFile(e) {
    const file = e.target.files?.[0]
    e.target.value = '' // allow re-selecting the same file later
    if (!file) return

    if (!file.name.toLowerCase().endsWith('.py')) {
      setLocalError(`"${file.name}" is not a .py file. Only Python source files are supported.`)
      return
    }

    const reader = new FileReader()
    reader.onload = () => {
      setSource(String(reader.result ?? ''))
      setFilename(file.name)
      setLocalError(null)
    }
    reader.onerror = () => setLocalError('Could not read the selected file.')
    reader.readAsText(file)
  }

  function handleSubmit(e) {
    e.preventDefault()
    if (loading) return
    if (!source.trim()) {
      setLocalError('Paste some Python code or upload a .py file first.')
      return
    }
    setLocalError(null)
    onAnalyze(source, filename.trim() || 'input.py')
  }

  const error = localError || apiError

  return (
    <main className="animate-fade-in mx-auto min-h-screen w-full max-w-5xl px-5 pb-10 pt-10 sm:px-8 lg:pt-16">
      <header className="mb-7 max-w-3xl sm:mb-9">
        <div className="mb-5 inline-flex items-center gap-2 border-l-2 pl-3" style={{ borderColor: COLORS.accent }}>
          <Network size={15} style={{ color: COLORS.accent }} />
          <span className="text-[11px] font-semibold uppercase" style={{ color: COLORS.textSecondary }}>
            AI-Powered Code Intelligence
          </span>
        </div>
        <h1
          className="text-4xl font-semibold leading-tight sm:text-[44px]"
          style={{ color: COLORS.textPrimary }}
        >
          RefactorGraph
        </h1>
        <p className="mt-3 max-w-2xl text-[15px] leading-7" style={{ color: COLORS.textSecondary }}>
          Detect Long Method, Feature Envy, and God Class from real code structure — then get a matching refactoring recommendation.
        </p>
        <div className="mt-5 flex flex-wrap gap-x-5 gap-y-2">
          {Object.entries(SMELL_LABELS).map(([task, label]) => (
            <span
              key={task}
              className="inline-flex items-center gap-2 text-xs font-medium"
              style={{ color: COLORS.textSecondary }}
            >
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: SMELL_CHIP_COLOR[task] }} />
              {label}
            </span>
          ))}
        </div>
      </header>

      <form
        onSubmit={handleSubmit}
        className="overflow-hidden rounded-lg border"
        style={{ backgroundColor: COLORS.surface, borderColor: COLORS.borderDefault, boxShadow: COLORS.shadowCard }}
      >
        <div className="flex flex-col gap-2 border-b px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5" style={{ borderColor: COLORS.borderDefault }}>
          <label htmlFor="filename" className="text-xs font-semibold uppercase" style={{ color: COLORS.textSecondary }}>
            Filename
          </label>
          <input
            id="filename"
            type="text"
            value={filename}
            onChange={(e) => setFilename(e.target.value)}
            className="w-full min-w-0 rounded-md border px-3 py-2 font-mono text-[13px] outline-none transition-shadow duration-150 sm:w-64"
            style={{ backgroundColor: COLORS.surface2, borderColor: COLORS.borderDefault, color: COLORS.textPrimary }}
            onFocus={(e) => {
              e.target.style.borderColor = COLORS.accent
              e.target.style.boxShadow = `0 0 0 3px ${COLORS.glowAccent}`
            }}
            onBlur={(e) => {
              e.target.style.borderColor = COLORS.borderDefault
              e.target.style.boxShadow = 'none'
            }}
            placeholder="input.py"
          />
        </div>

        <textarea
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder={PLACEHOLDER}
          spellCheck={false}
          rows={10}
          className="block h-[38vh] min-h-[220px] max-h-[420px] w-full resize-y px-4 py-4 font-mono text-[13px] leading-6 outline-none transition-shadow duration-150 placeholder:text-[#929AA8] sm:px-5"
          style={{ backgroundColor: COLORS.surface, color: COLORS.textPrimary }}
          onFocus={(e) => {
            e.target.style.boxShadow = `inset 0 0 0 1px ${COLORS.accent}`
          }}
          onBlur={(e) => {
            e.target.style.boxShadow = 'none'
          }}
        />

        <div className="flex flex-col gap-3 border-t px-4 py-3 sm:flex-row sm:items-center sm:px-5" style={{ borderColor: COLORS.borderDefault, backgroundColor: COLORS.surface2 }}>
          <Button type="button" variant="secondary" className="flex w-full items-center justify-center sm:w-auto" onClick={() => fileInputRef.current?.click()}>
            <span className="flex items-center gap-1.5">
              <Upload size={14} />
              Upload Python File
            </span>
          </Button>
          <input ref={fileInputRef} type="file" accept=".py" onChange={handleFile} className="hidden" />

          <Button type="submit" variant="primary" disabled={loading} className="flex w-full items-center justify-center font-semibold sm:ml-auto sm:w-auto">
            {loading ? (
              <span className="flex items-center gap-1.5">
                <Loader2 size={14} className="animate-spin" />
                Analyzing…
              </span>
            ) : (
              'Analyze Code'
            )}
          </Button>
        </div>

        {error && (
          <p
            role="alert"
            className="m-4 rounded-md border px-4 py-2.5 text-[13px] sm:m-5"
            style={{ borderColor: COLORS.dangerBorder, backgroundColor: COLORS.dangerBg, color: COLORS.danger }}
          >
            {error}
          </p>
        )}
      </form>
    </main>
  )
}
