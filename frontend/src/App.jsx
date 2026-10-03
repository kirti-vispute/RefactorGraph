import { useState } from 'react'
import { analyzeCode } from './api'
import CodeInputPage from './components/CodeInputPage'
import ResultsPage from './components/ResultsPage'

export default function App() {
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handleAnalyze(source, filename) {
    setLoading(true)
    setError(null)
    try {
      const result = await analyzeCode(source, filename)
      setAnalysis(result)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function handleReset() {
    setAnalysis(null)
    setError(null)
  }

  if (analysis) {
    return <ResultsPage analysis={analysis} onReset={handleReset} />
  }

  return <CodeInputPage onAnalyze={handleAnalyze} loading={loading} apiError={error} />
}
