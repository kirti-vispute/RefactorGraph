import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { MOCK_ANALYSIS } from '../../testFixtures'
import SmellDetailPanel from '../SmellDetailPanel'

describe('SmellDetailPanel', () => {
  it('prompts for a selection when nothing is selected', () => {
    render(<SmellDetailPanel selection={null} />)
    expect(screen.getByText(/select a detected smell/i)).toBeInTheDocument()
  })

  it('renders every real field from the selected prediction', () => {
    const selection = { task: 'long_method', prediction: MOCK_ANALYSIS.long_method[0] }
    render(<SmellDetailPanel selection={selection} />)

    expect(screen.getByText('Long Method')).toBeInTheDocument()
    expect(screen.getByText('91.0%')).toBeInTheDocument()
    expect(screen.getByText('school.py')).toBeInTheDocument()
    expect(screen.getByText('Course')).toBeInTheDocument()
    expect(screen.getByText('get_marks')).toBeInTheDocument()
    expect(screen.getByText('3–5')).toBeInTheDocument()
    expect(screen.getByText(MOCK_ANALYSIS.long_method[0].explanation)).toBeInTheDocument()
    expect(screen.getByText('Extract Method')).toBeInTheDocument()
    expect(screen.getByText(/does not automatically rewrite/i)).toBeInTheDocument()
  })

  it('falls back to "Not available from analysis." when class_name is missing', () => {
    const helper = MOCK_ANALYSIS.long_method[1] // "helper", class_name: null
    render(<SmellDetailPanel selection={{ task: 'long_method', prediction: helper }} />)
    expect(screen.getByText('Not available from analysis.')).toBeInTheDocument()
  })

  it('maps each smell to its project-defined refactoring recommendation', () => {
    const cases = [
      ['feature_envy', 'Move Method'],
      ['god_class', 'Split Class'],
    ]
    for (const [task, recommendation] of cases) {
      const prediction = MOCK_ANALYSIS[task][0]
      render(<SmellDetailPanel selection={{ task, prediction }} />)
      expect(screen.getByText(recommendation)).toBeInTheDocument()
    }
  })
})
