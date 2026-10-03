import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'
import { MOCK_ANALYSIS } from '../../testFixtures'
import CodeGraph from '../CodeGraph'

const TASKS = ['long_method', 'feature_envy', 'god_class']

function buildPredictionsByNodeId(analysis) {
  const map = new Map()
  for (const task of TASKS) {
    for (const prediction of analysis[task]) {
      const entries = map.get(prediction.node_id) ?? []
      entries.push({ task, prediction })
      map.set(prediction.node_id, entries)
    }
  }
  return map
}

async function renderGraph(extraProps = {}) {
  let cy
  render(
    <CodeGraph
      graph={MOCK_ANALYSIS.graph}
      predictionsByNodeId={buildPredictionsByNodeId(MOCK_ANALYSIS)}
      focusSelection={null}
      onReady={(instance) => { cy = instance }}
      {...extraProps}
    />,
  )
  await waitFor(() => expect(cy).toBeTruthy())
  return cy
}

describe('CodeGraph', () => {
  it('reports the real node/edge counts from the backend graph, not a guess', async () => {
    await renderGraph()
    expect(screen.getByText((_, node) => node?.textContent === '5 nodes · 3 edges · red ring = detected smell')).toBeInTheDocument()
  })

  it('only offers filters for node types actually present in the graph', async () => {
    await renderGraph()
    // MOCK_ANALYSIS.graph.nodes uses: module, class, method, function, parameter
    for (const type of ['module', 'class', 'method', 'function', 'parameter']) {
      expect(screen.getByRole('checkbox', { name: type })).toBeInTheDocument()
    }
    expect(screen.queryByRole('checkbox', { name: 'attribute' })).not.toBeInTheDocument()
    expect(screen.queryByRole('checkbox', { name: 'variable' })).not.toBeInTheDocument()
  })

  it('shows real node details and its associated detection when a node is tapped', async () => {
    const cy = await renderGraph()

    act(() => {
      cy.$id('method:0').emit('tap')
    })

    expect(await screen.findByText('Course.get_marks')).toBeInTheDocument()
    expect(screen.getByText('Long Method')).toBeInTheDocument()
    expect(screen.getByText(/91\.0% — detected/)).toBeInTheDocument()
  })

  it('shows "no prediction" for a node with no associated smell', async () => {
    const cy = await renderGraph()
    act(() => {
      cy.$id('parameter:0').emit('tap')
    })
    expect(await screen.findByText(/no smell prediction associated/i)).toBeInTheDocument()
  })

  it('shows relationship details when an edge is tapped', async () => {
    const cy = await renderGraph()
    const edgeId = cy.edges()[0].id()
    act(() => {
      cy.$id(edgeId).emit('tap')
    })
    expect(await screen.findByText('contains')).toBeInTheDocument()
  })

  it('toggling a legend entry off dims it and dims the matching nodes (not hides)', async () => {
    const cy = await renderGraph()
    const checkbox = screen.getByRole('checkbox', { name: 'function' })
    const label = checkbox.closest('label')

    await userEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
    expect(label).toHaveStyle({ color: 'rgb(77, 85, 98)' }) // COLORS.textDisabled
    expect(cy.$id('function:0').hasClass('dimmed')).toBe(true)

    await userEvent.click(checkbox)
    expect(checkbox).toBeChecked()
    expect(label).toHaveStyle({ color: 'rgb(167, 175, 188)' }) // COLORS.textSecondary
    expect(cy.$id('function:0').hasClass('dimmed')).toBe(false)
  })

  it('"Select All" checkbox toggles every node type at once', async () => {
    const cy = await renderGraph()
    const selectAll = screen.getByRole('checkbox', { name: /select all/i })
    expect(selectAll).toBeChecked()

    await userEvent.click(selectAll)
    expect(selectAll).not.toBeChecked()
    expect(cy.$id('function:0').hasClass('dimmed')).toBe(true)
    expect(cy.$id('method:0').hasClass('dimmed')).toBe(true)

    await userEvent.click(selectAll)
    expect(selectAll).toBeChecked()
    expect(cy.$id('function:0').hasClass('dimmed')).toBe(false)
  })

  it('reset view clears node selection', async () => {
    const cy = await renderGraph()
    act(() => {
      cy.$id('method:0').emit('tap')
    })
    expect(await screen.findByText('Course.get_marks')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: /^reset$/i }))
    expect(screen.getByText(/click a node or edge to inspect/i)).toBeInTheDocument()
  })
})
