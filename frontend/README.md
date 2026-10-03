# RefactorGraph — Frontend

React + Tailwind CSS + Cytoscape.js client for RefactorGraph. Submits
Python source to the backend's `/analyze` endpoint, renders the
returned code graph as an interactive Cytoscape visualization, and
displays each node's Long Method / Feature Envy / God Class predictions
alongside a matching refactoring recommendation.

## Stack

- **React** (function components, hooks) for the UI.
- **Tailwind CSS** for styling.
- **Cytoscape.js** (`cytoscape-fcose` layout) for the interactive code
  graph.
- **Vite** for dev server/build, **Vitest** + Testing Library for tests.

## Backend integration

The frontend never computes or re-thresholds predictions itself — it
sends raw source to the backend's `POST /analyze` and renders exactly
what comes back: the code graph, each node's Model Score (the model's
raw prediction output), the `predicted` boolean the backend already
thresholded, and the corresponding refactoring recommendation.

## Running it

```bash
npm install
npm run dev      # dev server, default http://localhost:5173
npm run build    # production build
npm run lint      # oxlint
```

Requires the backend running separately (default
`http://localhost:8000`) — see the project root `README.md`.

## Testing

```bash
npx vitest run
```
