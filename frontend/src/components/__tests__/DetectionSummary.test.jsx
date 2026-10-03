import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { MOCK_ANALYSIS } from '../../testFixtures'
import DetectionSummary from '../DetectionSummary'

describe('DetectionSummary', () => {
  it('counts only predicted=true detections per task, using real backend values', () => {
    render(<DetectionSummary results={MOCK_ANALYSIS} selected={null} onSelect={vi.fn()} />)

    // long_method has 2 entries but only 1 predicted=true
    const counts = screen.getAllByText(/^\d+$/).map((el) => el.textContent)
    expect(counts).toEqual(['1', '0', '1']) // long_method, feature_envy, god_class
  })

  it('renders real names and confidence for each detection, not raw indices', () => {
    render(<DetectionSummary results={MOCK_ANALYSIS} selected={null} onSelect={vi.fn()} />)

    expect(screen.getByText('Course.get_marks')).toBeInTheDocument()
    expect(screen.getByText('91.0%')).toBeInTheDocument()
    expect(screen.getByText('Course')).toBeInTheDocument()
    expect(screen.getByText('77.0%')).toBeInTheDocument()

    // helper (long_method, predicted=false) must not appear as a detection row
    expect(screen.queryByText('helper')).not.toBeInTheDocument()
  })

  it('calls onSelect with the task and full prediction object on click', async () => {
    const onSelect = vi.fn()
    render(<DetectionSummary results={MOCK_ANALYSIS} selected={null} onSelect={onSelect} />)

    await userEvent.click(screen.getByText('Course.get_marks'))

    expect(onSelect).toHaveBeenCalledWith('long_method', MOCK_ANALYSIS.long_method[0])
  })
})
