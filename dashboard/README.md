# GEcko research dashboard

This dashboard presents the main results from the GEcko research project. It is
a companion to the Python analysis, not a live trading interface.

**[Open the live dashboard](https://gecko-research-dashboard.greenwoodtom.chatgpt.site)**

## Content

- Candidate pair results and rejection reasons
- Rolling two-year stability checks
- Walk-forward OU signal construction
- Tax-only and realistic cost results
- Grand Exchange capacity scenarios
- OU and machine learning comparisons
- Lookahead audit results
- Research limits

The figures and reported values come from the committed outputs in the parent
repository. The dashboard does not fetch live prices. It also does not allow
users to tune strategy thresholds.

## Run locally

Node.js 22.13 or newer is required.

```bash
npm install
npm run dev
```

The local server uses `http://localhost:3000` by default.

## Test

```bash
npm run lint
npm test
```

The test command creates a production build and checks the rendered dashboard
content and assets.

## Main files

- `app/research-dashboard.tsx`: content, project values, and controls
- `app/globals.css`: layout and responsive styles
- `public/research/`: figures from the Python pipeline
- `tests/rendered-html.test.mjs`: production render checks
