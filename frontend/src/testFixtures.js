// Shape mirrors backend/app/schemas.py::AnalyzeResponse exactly -- used
// across component tests so a schema drift shows up as one fixture edit,
// not a silent mismatch scattered across files.
export const MOCK_ANALYSIS = {
  summary: {
    filename: 'school.py',
    n_classes: 1,
    n_methods: 1,
    n_functions: 1,
    n_nodes: 6,
    n_edges: 3,
    n_detected: 2,
  },
  graph: {
    nodes: [
      { id: 'module:0', type: 'module', name: 'school.py', line_start: 1, line_end: 10 },
      { id: 'class:0', type: 'class', name: 'Course', line_start: 2, line_end: 5 },
      { id: 'method:0', type: 'method', name: 'Course.get_marks', line_start: 3, line_end: 5 },
      { id: 'function:0', type: 'function', name: 'helper', line_start: 7, line_end: 8 },
      { id: 'parameter:0', type: 'parameter', name: 'Course.get_marks.self', line_start: null, line_end: null },
    ],
    edges: [
      { source: 'module:0', target: 'class:0', type: 'contains' },
      { source: 'class:0', target: 'method:0', type: 'contains' },
      { source: 'module:0', target: 'function:0', type: 'contains' },
    ],
  },
  long_method: [
    {
      node_id: 'method:0', name: 'Course.get_marks', node_type: 'method', probability: 0.91, predicted: true,
      file: 'school.py', class_name: 'Course', line_start: 3, line_end: 5,
      explanation: 'loc=3, statements=1, params=1, self_access=0, external_access=0',
    },
    {
      node_id: 'function:0', name: 'helper', node_type: 'function', probability: 0.12, predicted: false,
      file: 'school.py', class_name: null, line_start: 7, line_end: 8,
      explanation: 'loc=2, statements=1, params=0, self_access=0, external_access=0',
    },
  ],
  feature_envy: [
    {
      node_id: 'method:0', name: 'Course.get_marks', node_type: 'method', probability: 0.05, predicted: false,
      file: 'school.py', class_name: 'Course', line_start: 3, line_end: 5,
      explanation: 'loc=3, statements=1, params=1, self_access=0, external_access=0',
    },
  ],
  god_class: [
    {
      node_id: 'class:0', name: 'Course', node_type: 'class', probability: 0.77, predicted: true,
      file: 'school.py', class_name: 'Course', line_start: 2, line_end: 5,
      explanation: 'loc=4, methods=1, fields=0',
    },
  ],
}
