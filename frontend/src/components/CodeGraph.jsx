import cytoscape from 'cytoscape'
import fcose from 'cytoscape-fcose'
import { Maximize2, RotateCcw, Search } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  COLORS,
  EDGE_TYPE_COLORS,
  NODE_TYPE_COLORS,
  NOT_AVAILABLE,
  REFACTORING_RECOMMENDATIONS,
  SMELL_LABELS,
} from '../constants'

cytoscape.use(fcose)

// Visual language matches Graphify (github.com/Graphify-Labs/graphify)'s
// graph.html: small force-clustered dots sized by degree, labels shown only
// for the handful of highest-degree "hub" nodes (not every node -- with 30+
// nodes, permanent labels on everything is unreadable), thin translucent
// edges, and a checkbox legend that toggles whole categories. We color by
// structural node TYPE rather than a detected "community" since
// graph_builder.py does no community detection -- RefactorGraph's own
// addition (red ring = a real backend-detected smell, click-through to the
// actual prediction) is layered on top rather than replacing this.
const MIN_SIZE = 6
const MAX_SIZE = 34

function stylesheet() {
  const nodeColorRules = Object.entries(NODE_TYPE_COLORS).map(([type, color]) => ({
    selector: `node[type = "${type}"]`,
    style: { 'background-color': color },
  }))
  const edgeColorRules = Object.entries(EDGE_TYPE_COLORS).map(([type, color]) => ({
    selector: `edge[type = "${type}"]`,
    style: { 'line-color': color },
  }))

  return [
    {
      selector: 'node',
      style: {
        shape: 'ellipse',
        width: 'data(size)',
        height: 'data(size)',
        label: 'data(label)',
        color: COLORS.textPrimary,
        'font-family': 'var(--font-mono)',
        'font-size': 'data(fontSize)',
        'font-weight': 500,
        'text-valign': 'bottom',
        'text-margin-y': 3,
        'text-outline-width': 2,
        'text-outline-color': COLORS.bg,
        'background-opacity': 0.95,
        'border-width': 1.5,
        'border-color': COLORS.bg,
        'border-opacity': 0.8,
        'shadow-blur': 10,
        'shadow-color': 'black',
        'shadow-opacity': 0.45,
        'shadow-offset-x': 0,
        'shadow-offset-y': 1,
        'transition-property': 'shadow-blur, shadow-opacity, border-width, border-color, border-opacity',
        'transition-duration': 150,
      },
    },
    ...nodeColorRules,
    {
      selector: 'node.hovered',
      style: {
        'border-width': 2,
        'border-color': COLORS.textPrimary,
        'border-opacity': 0.95,
        'shadow-blur': 18,
        'shadow-color': COLORS.textPrimary,
        'shadow-opacity': 0.25,
      },
    },
    {
      selector: 'node.smell',
      style: {
        'border-width': 2.5,
        'border-color': COLORS.danger,
        'border-opacity': 1,
        'shadow-blur': 22,
        'shadow-color': COLORS.danger,
        'shadow-opacity': 0.55,
      },
    },
    {
      selector: 'node.focused',
      style: {
        'border-width': 3,
        'border-color': COLORS.accent,
        'border-opacity': 1,
        'shadow-blur': 24,
        'shadow-color': COLORS.accent,
        'shadow-opacity': 0.6,
      },
    },
    {
      selector: 'node.search-match',
      style: {
        'border-width': 3,
        'border-color': COLORS.warning,
        'border-opacity': 1,
        'shadow-blur': 24,
        'shadow-color': COLORS.warning,
        'shadow-opacity': 0.6,
      },
    },
    {
      selector: 'node.dimmed, edge.dimmed',
      style: { opacity: 0.1 },
    },
    {
      selector: 'edge',
      style: {
        width: 1,
        'line-color': '#4a5568',
        'curve-style': 'haystack',
        'haystack-radius': 0.3,
        opacity: 0.32,
        label: '',
        'line-cap': 'round',
        'transition-property': 'opacity, width',
        'transition-duration': 150,
      },
    },
    {
      selector: 'edge.label-visible',
      style: {
        'curve-style': 'bezier',
        opacity: 0.95,
        width: 1.6,
        label: 'data(type)',
        'font-family': 'var(--font-mono)',
        'font-size': 9,
        color: COLORS.textSecondary,
        'text-rotation': 'autorotate',
        'text-background-color': COLORS.surface2,
        'text-background-opacity': 0.95,
        'text-background-padding': 3,
        'text-background-shape': 'roundrectangle',
        'text-border-width': 1,
        'text-border-color': COLORS.borderDefault,
        'text-border-opacity': 1,
        'target-arrow-shape': 'triangle',
        'target-arrow-color': '#4a5568',
        'arrow-scale': 0.7,
        'line-cap': 'round',
      },
    },
    ...edgeColorRules,
    ...Object.entries(EDGE_TYPE_COLORS).map(([type, color]) => ({
      selector: `edge[type = "${type}"].label-visible`,
      style: { 'target-arrow-color': color },
    })),
    {
      selector: 'edge.focused',
      style: {
        'curve-style': 'bezier',
        'line-fill': 'linear-gradient',
        'line-gradient-stop-colors': `${COLORS.accent} ${COLORS.accentHover}`,
        'line-gradient-stop-positions': '0% 100%',
        'line-color': COLORS.accent,
        width: 2,
        opacity: 1,
        label: 'data(type)',
        color: COLORS.textPrimary,
        'target-arrow-shape': 'triangle',
        'target-arrow-color': COLORS.accentHover,
        'arrow-scale': 0.8,
        'line-cap': 'round',
      },
    },
  ]
}

export default function CodeGraph({ graph, predictionsByNodeId, focusSelection, onReady }) {
  const containerRef = useRef(null)
  const cyRef = useRef(null)
  const [activeTypes, setActiveTypes] = useState(() => new Set())
  const [search, setSearch] = useState('')
  const [selectedNodeId, setSelectedNodeId] = useState(null)
  const [selectedEdge, setSelectedEdge] = useState(null)
  const [tooltip, setTooltip] = useState(null) // { x, y, node }

  const presentTypes = useMemo(() => {
    const types = new Set(graph.nodes.map((n) => n.type))
    return [...types]
  }, [graph])

  const typeCounts = useMemo(() => {
    const counts = new Map()
    for (const n of graph.nodes) counts.set(n.type, (counts.get(n.type) ?? 0) + 1)
    return counts
  }, [graph])

  const nodeById = useMemo(() => new Map(graph.nodes.map((n) => [n.id, n])), [graph])

  // Real degree from the actual returned edges -- drives node size and
  // which nodes get a permanent label, same idea Graphify's graph.html uses
  // for its community graph.
  const { degreeById, maxDegree } = useMemo(() => {
    const deg = new Map(graph.nodes.map((n) => [n.id, 0]))
    for (const e of graph.edges) {
      deg.set(e.source, (deg.get(e.source) ?? 0) + 1)
      deg.set(e.target, (deg.get(e.target) ?? 0) + 1)
    }
    return { degreeById: deg, maxDegree: Math.max(1, ...deg.values()) }
  }, [graph])

  const elements = useMemo(
    () => [
      ...graph.nodes.map((n) => {
        const degree = degreeById.get(n.id) ?? 0
        const size = MIN_SIZE + (MAX_SIZE - MIN_SIZE) * (degree / maxDegree)
        const labelVisible = degree >= 0.35 * maxDegree
        const shortName = n.type === 'module' || n.type === 'import' ? n.name : n.name.split('.').pop()
        return {
          data: {
            id: n.id,
            type: n.type,
            name: n.name,
            degree,
            size,
            fontSize: labelVisible ? 10 : 0,
            label: labelVisible ? shortName : '',
          },
        }
      }),
      ...graph.edges.map((e, i) => ({
        data: { id: `e${i}`, source: e.source, target: e.target, type: e.type },
      })),
    ],
    [graph, degreeById, maxDegree],
  )

  useEffect(() => {
    const cy = cytoscape({
      // jsdom (used by the test suite) has no 2d canvas context, so
      // cytoscape's canvas renderer can't initialize there -- headless mode
      // is cytoscape's own supported mode for exactly this (full graph
      // model + events, no rendering), so tests exercise real graph/event
      // logic instead of a mock. The real app always renders normally.
      headless: import.meta.env.MODE === 'test',
      container: containerRef.current,
      elements,
      style: stylesheet(),
      layout: {
        name: 'fcose',
        quality: 'proof',
        animate: false,
        nodeSeparation: 60,
        idealEdgeLength: 55,
        nodeRepulsion: 6500,
        gravity: 0.3,
        padding: 30,
        fit: true,
      },
      wheelSensitivity: 0.25,
      minZoom: 0.1,
      maxZoom: 4,
    })

    cy.on('tap', 'node', (evt) => {
      setSelectedEdge(null)
      setSelectedNodeId(evt.target.id())
    })
    cy.on('tap', 'edge', (evt) => {
      setSelectedNodeId(null)
      setSelectedEdge(evt.target.data())
    })
    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        setSelectedNodeId(null)
        setSelectedEdge(null)
      }
    })

    cy.on('mouseover', 'node', (evt) => {
      const node = evt.target
      node.addClass('hovered')
      node.connectedEdges().addClass('label-visible')
      const pos = node.renderedPosition()
      const rect = containerRef.current.getBoundingClientRect()
      setTooltip({ x: rect.left + pos.x, y: rect.top + pos.y, node: nodeById.get(node.id()) })
    })
    cy.on('mouseout', 'node', (evt) => {
      const node = evt.target
      node.removeClass('hovered')
      node.connectedEdges().removeClass('label-visible')
      setTooltip(null)
    })
    cy.on('mouseover', 'edge', (evt) => evt.target.addClass('label-visible'))
    cy.on('mouseout', 'edge', (evt) => evt.target.removeClass('label-visible'))

    cyRef.current = cy
    setActiveTypes(new Set(presentTypes))
    onReady?.(cy)

    // Cytoscape sizes its canvas from the container's dimensions ONCE at
    // construction time -- it has no built-in awareness of later container
    // resizes (a responsive breakpoint change, a window resize, a sidebar
    // toggling). Without this, the canvas silently keeps its original pixel
    // size while the container div shrinks/grows around it, so the graph
    // visibly overflows or leaves dead space instead of staying fitted.
    let resizeObserver
    if (containerRef.current && typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver(() => {
        cy.resize()
        cy.fit(undefined, 40)
      })
      resizeObserver.observe(containerRef.current)
    }

    return () => {
      resizeObserver?.disconnect()
      cy.destroy()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().removeClass('smell')
    for (const nodeId of predictionsByNodeId.keys()) {
      const hasDetection = predictionsByNodeId.get(nodeId).some((entry) => entry.prediction.predicted)
      if (hasDetection) cy.$id(nodeId).addClass('smell')
    }
  }, [predictionsByNodeId])

  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return
    cy.nodes().forEach((n) => n.toggleClass('dimmed', !activeTypes.has(n.data('type'))))
    cy.edges().forEach((e) => {
      const dimmed = e.source().hasClass('dimmed') || e.target().hasClass('dimmed')
      e.toggleClass('dimmed', dimmed)
    })
  }, [activeTypes])

  const searchMatches = useMemo(() => {
    if (!search.trim()) return []
    const term = search.trim().toLowerCase()
    return graph.nodes.filter((n) => n.name.toLowerCase().includes(term)).slice(0, 20)
  }, [graph, search])

  function focusNode(nodeId) {
    const cy = cyRef.current
    if (!cy) return
    const node = cy.$id(nodeId)
    if (node.empty()) return
    setSelectedNodeId(nodeId)
    setSelectedEdge(null)
    cy.elements().removeClass('search-match focused dimmed label-visible')
    node.addClass('search-match')
    cy.animate({ center: { eles: node }, zoom: 1.8 }, { duration: 250 })
  }

  useEffect(() => {
    const cy = cyRef.current
    if (!cy || !focusSelection) return
    const { node_id: nodeId } = focusSelection.prediction
    const node = cy.$id(nodeId)
    if (node.empty()) return

    setSelectedNodeId(nodeId)
    setSelectedEdge(null)

    const neighborhood = node.closedNeighborhood()
    cy.elements().removeClass('focused search-match label-visible')
    cy.elements().addClass('dimmed')
    neighborhood.removeClass('dimmed')
    neighborhood.addClass('focused')
    neighborhood.edges().addClass('label-visible')
    cy.animate({ fit: { eles: neighborhood, padding: 80 } }, { duration: 250 })
  }, [focusSelection])

  function fit() {
    cyRef.current?.fit(undefined, 40)
  }

  function resetView() {
    const cy = cyRef.current
    cy?.elements().removeClass('dimmed focused search-match label-visible')
    setSearch('')
    setActiveTypes(new Set(presentTypes))
    setSelectedNodeId(null)
    setSelectedEdge(null)
    fit()
  }

  function toggleType(type) {
    setActiveTypes((prev) => {
      const next = new Set(prev)
      if (next.has(type)) next.delete(type)
      else next.add(type)
      return next
    })
  }

  function toggleAll() {
    setActiveTypes((prev) => (prev.size === presentTypes.length ? new Set() : new Set(presentTypes)))
  }

  const selectedNode = selectedNodeId ? nodeById.get(selectedNodeId) : null
  const selectedPredictions = selectedNodeId ? predictionsByNodeId.get(selectedNodeId) ?? [] : []
  const neighborNames = useMemo(() => {
    if (!selectedNodeId) return []
    const names = new Set()
    for (const e of graph.edges) {
      if (e.source === selectedNodeId) names.add(nodeById.get(e.target)?.name)
      if (e.target === selectedNodeId) names.add(nodeById.get(e.source)?.name)
    }
    return [...names].filter(Boolean)
  }, [selectedNodeId, graph, nodeById])

  const allSelected = activeTypes.size === presentTypes.length

  return (
    <section>
      <div className="mb-3 flex items-baseline justify-between">
        <div>
          <h2 className="text-[15px] font-semibold text-[color:var(--text-primary)]">Code Structure Graph</h2>
          <p className="mt-0.5 text-xs text-[color:var(--text-muted)]">
            {graph.nodes.length} nodes &middot; {graph.edges.length} edges &middot; red ring = detected smell
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1fr_260px]">
        <div className="relative">
          <div
            ref={containerRef}
            className="h-[520px] rounded-[10px] border"
            style={{
              backgroundColor: COLORS.bg,
              backgroundImage: `radial-gradient(ellipse 70% 60% at 50% 40%, ${COLORS.surface2} 0%, ${COLORS.bg} 100%)`,
              borderColor: COLORS.borderDefault,
              boxShadow: COLORS.shadowCard,
            }}
          />
          <div className="absolute right-3 top-3 flex gap-1.5">
            <button
              type="button"
              onClick={fit}
              title="Fit graph to view"
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium shadow-sm backdrop-blur transition-all hover:-translate-y-0.5"
              style={{ backgroundColor: `${COLORS.surface2}e6`, borderColor: COLORS.borderDefault, color: COLORS.textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = COLORS.accent; e.currentTarget.style.color = COLORS.textPrimary }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = COLORS.borderDefault; e.currentTarget.style.color = COLORS.textSecondary }}
            >
              <Maximize2 size={13} />
              Fit
            </button>
            <button
              type="button"
              onClick={resetView}
              title="Reset selection and view"
              className="flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium shadow-sm backdrop-blur transition-all hover:-translate-y-0.5"
              style={{ backgroundColor: `${COLORS.surface2}e6`, borderColor: COLORS.borderDefault, color: COLORS.textSecondary }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = COLORS.accent; e.currentTarget.style.color = COLORS.textPrimary }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = COLORS.borderDefault; e.currentTarget.style.color = COLORS.textSecondary }}
            >
              <RotateCcw size={13} />
              Reset
            </button>
          </div>

          {tooltip && (
            <div
              className="pointer-events-none fixed z-20 -translate-x-1/2 -translate-y-[calc(100%+10px)] rounded-md border px-2.5 py-1.5 text-xs shadow-lg"
              style={{ left: tooltip.x, top: tooltip.y, backgroundColor: COLORS.elevated, borderColor: COLORS.borderStrong }}
            >
              <div className="font-mono font-medium" style={{ color: COLORS.textPrimary }}>{tooltip.node.name}</div>
              <div className="capitalize" style={{ color: COLORS.textMuted }}>{tooltip.node.type}</div>
            </div>
          )}
        </div>

        <div className="rounded-[10px] border p-3 text-sm" style={{ backgroundColor: COLORS.surface, borderColor: COLORS.borderDefault, boxShadow: COLORS.shadowCard }}>
          <div className="relative">
            <Search size={13} className="pointer-events-none absolute left-2.5 top-1/2 -translate-y-1/2" style={{ color: COLORS.textMuted }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search nodes..."
              className="w-full rounded-md border py-1.5 pl-7 pr-2.5 text-xs outline-none transition-colors"
              style={{ backgroundColor: COLORS.bg, borderColor: COLORS.borderDefault, color: COLORS.textPrimary }}
              onFocus={(e) => (e.target.style.borderColor = COLORS.accent)}
              onBlur={(e) => (e.target.style.borderColor = COLORS.borderDefault)}
            />
            {searchMatches.length > 0 && (
              <ul
                className="absolute z-10 mt-1 max-h-[140px] w-full overflow-y-auto rounded-md border shadow-lg"
                style={{ backgroundColor: COLORS.elevated, borderColor: COLORS.borderStrong }}
              >
                {searchMatches.map((n) => (
                  <li key={n.id}>
                    <button
                      type="button"
                      onClick={() => focusNode(n.id)}
                      className="block w-full truncate border-l-[3px] px-2 py-1 text-left font-mono text-xs transition-colors"
                      style={{ borderColor: NODE_TYPE_COLORS[n.type], color: COLORS.textSecondary }}
                    >
                      {n.name}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="mt-3 border-t pt-3" style={{ borderColor: COLORS.divider }}>
            {!selectedNode && !selectedEdge && (
              <p className="text-xs" style={{ color: COLORS.textMuted }}>Click a node or edge to inspect it.</p>
            )}

            {selectedEdge && (
              <div className="space-y-1">
                <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>Relationship</div>
                <div className="truncate font-mono text-xs" style={{ color: COLORS.textSecondary }}>{selectedEdge.source}</div>
                <div className="text-center text-xs font-medium" style={{ color: COLORS.accent }}>{selectedEdge.type}</div>
                <div className="truncate font-mono text-xs" style={{ color: COLORS.textSecondary }}>{selectedEdge.target}</div>
              </div>
            )}

            {selectedNode && (
              <div className="space-y-2.5">
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>Node Type</div>
                  <div className="capitalize" style={{ color: COLORS.textPrimary }}>{selectedNode.type}</div>
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>Name</div>
                  <div className="break-all font-mono" style={{ color: COLORS.textPrimary }}>{selectedNode.name}</div>
                </div>
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>Source Lines</div>
                  <div style={{ color: COLORS.textPrimary }}>
                    {selectedNode.line_start != null ? `${selectedNode.line_start}–${selectedNode.line_end}` : NOT_AVAILABLE}
                  </div>
                </div>

                {neighborNames.length > 0 && (
                  <div>
                    <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>
                      Neighbors ({neighborNames.length})
                    </div>
                    <ul className="mt-1 max-h-[140px] space-y-0.5 overflow-y-auto font-mono text-xs" style={{ color: COLORS.textSecondary }}>
                      {neighborNames.map((name) => (
                        <li key={name} className="truncate">{name}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {selectedPredictions.length === 0 && (
                  <div className="pt-1 text-xs" style={{ color: COLORS.textMuted }}>No smell prediction associated with this node.</div>
                )}
                {selectedPredictions.map(({ task, prediction }) => (
                  <div key={task} className="rounded-md border p-2" style={{ backgroundColor: COLORS.bg, borderColor: COLORS.borderDefault }}>
                    <div className="text-xs font-semibold" style={{ color: COLORS.textSecondary }}>{SMELL_LABELS[task]}</div>
                    <div className="text-xs" style={{ color: COLORS.textMuted }}>
                      Model Score {(prediction.probability * 100).toFixed(1)}% —{' '}
                      {prediction.predicted ? 'detected' : 'not detected'}
                    </div>
                    {prediction.predicted && (
                      <div className="mt-1 text-xs font-medium" style={{ color: COLORS.accent }}>
                        → {REFACTORING_RECOMMENDATIONS[task]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="mt-3 border-t pt-3" style={{ borderColor: COLORS.divider }}>
            <div className="mb-1.5 flex items-center justify-between">
              <div className="text-[10px] font-semibold uppercase tracking-wide" style={{ color: COLORS.textMuted }}>Node Types</div>
            </div>
            <label className="mb-1 flex w-full cursor-pointer items-center gap-2 rounded px-1 py-1 text-xs font-medium" style={{ color: COLORS.textSecondary }}>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={toggleAll}
                className="h-3 w-3 accent-[--accent]"
                style={{ accentColor: COLORS.accent }}
              />
              Select All
            </label>
            <ul className="space-y-0.5">
              {presentTypes.map((type) => {
                const active = activeTypes.has(type)
                return (
                  <li key={type}>
                    <label
                      className="flex w-full cursor-pointer items-center gap-2 rounded px-1 py-1 text-left text-xs capitalize transition-colors"
                      style={{ color: active ? COLORS.textSecondary : COLORS.textDisabled }}
                    >
                      <input
                        type="checkbox"
                        aria-label={type}
                        checked={active}
                        onChange={() => toggleType(type)}
                        className="h-3 w-3"
                        style={{ accentColor: NODE_TYPE_COLORS[type] }}
                      />
                      <span
                        className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                        style={{ backgroundColor: NODE_TYPE_COLORS[type], opacity: active ? 1 : 0.35 }}
                      />
                      <span className="truncate">{type}</span>
                      <span className="ml-auto" style={{ color: COLORS.textMuted }}>{typeCounts.get(type)}</span>
                    </label>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
