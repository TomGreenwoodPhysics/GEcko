"use client";

import { useMemo, useState } from "react";
import Image from "next/image";

type Pair = {
  name: string;
  route: string;
  status: "Lead" | "Secondary" | "Rejected";
  p: string;
  halfLife: string;
  cv: string;
  rolling: string;
  note: string;
};

const pairs: Pair[] = [
  { name: "Hide", route: "Cowhide → Leather", status: "Lead", p: "1.96e−21", halfLife: "13.1d", cv: "9.0%", rolling: "99%", note: "The only relationship that stayed stable enough to carry into a cost-aware walk-forward test." },
  { name: "Rune", route: "Rune ore → Rune bar", status: "Secondary", p: "0.164", halfLife: "32.4d", cv: "2.7%", rolling: "79%", note: "Strong rolling stability after 2019, but the full-sample Engle–Granger result was inconclusive." },
  { name: "Iron", route: "Iron ore → Iron bar", status: "Rejected", p: "0.00038", halfLife: "67.0d", cv: "29.7%", rolling: "52%", note: "An attractive full-sample statistic that weakened materially under rolling-window inspection." },
  { name: "Silver", route: "Silver ore → Silver bar", status: "Rejected", p: "8.65e−11", halfLife: "53.1d", cv: "27.8%", rolling: "—", note: "Stationary ratio, but slower reversion and weaker economics than the lead transformation." },
  { name: "Steel", route: "Coal → Steel bar", status: "Rejected", p: "0.0042", halfLife: "51.1d", cv: "21.5%", rolling: "—", note: "The simplified screen omits iron ore, so the economic mapping is not a clean one-input transformation." },
  { name: "Mithril", route: "Mithril ore → Bar", status: "Rejected", p: "0.0090", halfLife: "151d", cv: "28.0%", rolling: "—", note: "Long half-life and omitted coal input made the initial relationship unsuitable for execution research." },
  { name: "Adamant", route: "Adamantite ore → Bar", status: "Rejected", p: "0.521", halfLife: "116d", cv: "19.8%", rolling: "—", note: "No convincing stationarity in the screened ratio and a slow estimated reversion speed." },
  { name: "Gold", route: "Gold ore → Gold bar", status: "Rejected", p: "0.340", halfLife: "647d", cv: "—", rolling: "—", note: "A multi-year half-life makes the relationship impractical for the intended rotation strategy." },
];

const costModels = {
  realistic: {
    label: "Tax + measured spread",
    return: "1,455%",
    annual: "35.1%",
    sharpe: "0.80",
    sortino: "1.18",
    drawdown: "−59.0%",
    rotations: "34",
    color: "var(--mint)",
    width: "36%",
  },
  tax: {
    label: "Tax only",
    return: "4,079%",
    annual: "50.6%",
    sharpe: "1.16",
    sortino: "1.73",
    drawdown: "−57.8%",
    rotations: "34",
    color: "var(--gold)",
    width: "52%",
  },
};

const capacity = [
  { capital: "728k", windows: "0%", max: "1", verdict: "Inside estimated capacity" },
  { capital: "1m", windows: "12%", max: "2", verdict: "Occasional queueing" },
  { capital: "10m", windows: "100%", max: "14", verdict: "Execution dominated" },
  { capital: "50m", windows: "100%", max: "69", verdict: "Not operationally credible" },
  { capital: "100m", windows: "100%", max: "138", verdict: "Outside tested capacity" },
];

const mlModels = [
  { name: "OU z-score", annual: 39.1, sharpe: 0.89, dd: "−59.0%", rotations: 25, dsr: "86.7%" },
  { name: "Logistic", annual: 17.0, sharpe: 0.37, dd: "−74.3%", rotations: 49, dsr: "52.3%" },
  { name: "Boosted trees", annual: -90.8, sharpe: -1.42, dd: "−100.0%", rotations: 501, dsr: "≈0%" },
];

function Arrow() {
  return <span aria-hidden="true">↗</span>;
}

export function ResearchDashboard() {
  const [selectedPair, setSelectedPair] = useState("Hide");
  const [costMode, setCostMode] = useState<keyof typeof costModels>("realistic");
  const [capacityIndex, setCapacityIndex] = useState(0);

  const pair = useMemo(
    () => pairs.find((candidate) => candidate.name === selectedPair) ?? pairs[0],
    [selectedPair],
  );
  const performance = costModels[costMode];
  const capacityPoint = capacity[capacityIndex];

  return (
    <main>
      <nav className="nav shell" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="GEcko dashboard home">
          <span>GEcko</span>
          <small>Quantitative research</small>
        </a>
        <div className="nav-links">
          <a href="#evidence">Evidence</a>
          <a href="#economics">Economics</a>
          <a href="#method">Method</a>
        </div>
        <a className="source-link" href="https://github.com/TomGreenwoodPhysics/GEcko" target="_blank" rel="noreferrer">
          View source <Arrow />
        </a>
      </nav>

      <section className="hero shell" id="top">
        <div className="hero-copy">
          <p className="eyebrow">Research dashboard · Data through 20 June 2026</p>
          <h1>Grand Exchange transformation research</h1>
          <p className="hero-deck">
            A walk-forward study of relative value between economically linked Old School RuneScape items. Candidate relationships are screened for stability, then evaluated after tax, observed spreads and Grand Exchange limits.
          </p>
          <div className="hero-actions">
            <a className="button primary" href="#economics">View findings</a>
            <a className="button ghost" href="#method">Methodology</a>
          </div>
        </div>
        <aside className="verdict-card">
          <div className="verdict-head">
            <span className="tag positive">Primary relationship</span>
            <span className="micro">WALK-FORWARD TEST</span>
          </div>
          <div className="pair-route">
            <span>Cowhide</span><span className="route-line" /><span>Leather</span>
          </div>
          <p>The hide–leather relationship passed the 5% threshold in <strong>99%</strong> of rolling two-year windows.</p>
          <div className="verdict-grid">
            <div><span>Realistic annual return</span><strong>35.1%</strong></div>
            <div><span>Deflated Sharpe probability</span><strong>89.8%</strong></div>
            <div><span>Maximum drawdown</span><strong>−59.0%</strong></div>
            <div><span>Estimated safe capital</span><strong>728k gp</strong></div>
          </div>
          <p className="honesty">Interpret with caution: maximum drawdown was 59% and estimated safe starting capital was 728k gp.</p>
        </aside>
      </section>

      <section className="ticker" aria-label="Study summary">
        <div className="ticker-inner shell">
          <span><b>4,000+</b> daily observations</span>
          <span><b>8</b> candidates</span>
          <span><b>111</b> rolling windows</span>
          <span><b>11.2 years</b> of data</span>
          <span><b>0</b> lookahead violations</span>
          <span className="as-of">Daily close data</span>
        </div>
      </section>

      <section className="section shell" id="evidence">
        <div className="section-heading">
          <div><p className="kicker">01 · Candidate screening</p><h2>Most relationships did not survive rolling analysis.</h2></div>
          <p>The process applies progressively stricter tests. A low full-sample p-value is treated as a diagnostic result, not evidence of a tradable strategy.</p>
        </div>

        <div className="funnel">
          <article><span className="stage">01</span><p>Economic screen</p><strong>8 pairs</strong><small>Linked production transformations</small></article>
          <span className="funnel-arrow" aria-hidden="true">→</span>
          <article><span className="stage">02</span><p>Time-series diagnostics</p><strong>3 candidates</strong><small>Hide, iron and rune</small></article>
          <span className="funnel-arrow" aria-hidden="true">→</span>
          <article><span className="stage">03</span><p>Rolling stability</p><strong>3 tested</strong><small>111 trailing windows each</small></article>
          <span className="funnel-arrow" aria-hidden="true">→</span>
          <article className="funnel-winner"><span className="stage">04</span><p>Cost-aware backtest</p><strong>1 primary result</strong><small>Hide ↔ leather</small></article>
        </div>

        <div className="pair-workbench">
          <div className="pair-picker" role="tablist" aria-label="Candidate pairs">
            {pairs.map((item) => (
              <button
                key={item.name}
                type="button"
                role="tab"
                aria-selected={selectedPair === item.name}
                className={selectedPair === item.name ? "active" : ""}
                onClick={() => setSelectedPair(item.name)}
              >
                <span>{item.name}</span>
                <small className={item.status.toLowerCase()}>{item.status}</small>
              </button>
            ))}
          </div>
          <article className="pair-detail" role="tabpanel">
            <div className="pair-detail-top">
              <div><p className="micro">SELECTED RELATIONSHIP</p><h3>{pair.route}</h3></div>
              <span className={`tag ${pair.status === "Lead" ? "positive" : pair.status === "Secondary" ? "caution" : "muted"}`}>{pair.status}</span>
            </div>
            <div className="metric-row">
              <div><span>ADF p-value</span><strong>{pair.p}</strong></div>
              <div><span>Half-life</span><strong>{pair.halfLife}</strong></div>
              <div><span>Ratio CV</span><strong>{pair.cv}</strong></div>
              <div><span>Rolling pass rate</span><strong>{pair.rolling}</strong></div>
            </div>
            <p className="pair-note">{pair.note}</p>
          </article>
        </div>
      </section>

      <section className="section contrast" id="stability">
        <div className="shell">
          <div className="section-heading">
            <div><p className="kicker">02 · Rolling stability</p><h2>Stability varies materially by pair.</h2></div>
            <p>Each line repeats Engle–Granger on a trailing 730-day window, advancing roughly monthly. The 5% threshold is fixed before inspection.</p>
          </div>
          <div className="stability-layout">
            <div className="figure-card">
              <Image width={1428} height={1174} src="/research/rolling-cointegration.png" alt="Rolling two-year Engle-Granger p-values for hide, iron, and rune pairs from 2017 through 2026" />
              <p className="figure-caption"><span>Figure 1</span> Rolling two-year cointegration diagnostics. Lower than 0.05 passes the window.</p>
            </div>
            <div className="finding-stack">
              <article className="finding featured"><span>99%</span><div><strong>Hide passes almost every window</strong><p>Rolling β stayed between 0.80 and 1.07, supporting a durable relative-price relationship.</p></div></article>
              <article className="finding"><span>79%</span><div><strong>Rune improves after 2019</strong><p>Promising, but its full-sample Engle–Granger p-value remains 0.34.</p></div></article>
              <article className="finding"><span>52%</span><div><strong>Iron is regime-dependent</strong><p>The headline p-value looked strong; rolling tests revealed extended periods of failure.</p></div></article>
              <p className="analysis-note"><strong>Interpretation</strong> Rolling-window consistency is given more weight than a single full-sample test.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="section shell" id="economics">
        <div className="section-heading">
          <div><p className="kicker">03 · Backtest economics</p><h2>Transaction costs reduce, but do not eliminate, the result.</h2></div>
          <p>Long-only rotation switches between raw and processed items at ±1.5 z-score with hysteresis. Every switch pays the 2% tax plus measured half-spreads.</p>
        </div>

        <div className="economics-grid">
          <article className="performance-card">
            <div className="toggle" aria-label="Backtest cost model">
              <button className={costMode === "realistic" ? "active" : ""} onClick={() => setCostMode("realistic")}>Realistic costs</button>
              <button className={costMode === "tax" ? "active" : ""} onClick={() => setCostMode("tax")}>Tax only</button>
            </div>
            <div className="performance-hero">
              <p>{performance.label}</p>
              <strong>{performance.return}</strong>
              <span>cumulative return · Apr 2017–Jun 2026</span>
            </div>
            <div className="metric-row performance-metrics">
              <div><span>Annualised</span><strong>{performance.annual}</strong></div>
              <div><span>Sharpe</span><strong>{performance.sharpe}</strong></div>
              <div><span>Sortino</span><strong>{performance.sortino}</strong></div>
              <div><span>Max drawdown</span><strong>{performance.drawdown}</strong></div>
            </div>
            <div className="return-scale" aria-label={`${performance.label} relative return scale`}>
              <span style={{ width: performance.width, background: performance.color }} />
            </div>
            <div className="cost-note"><b>{performance.rotations}</b> rotations <span /> Cowhide half-spread <b>0.85%</b> <span /> Leather half-spread <b>2.05%</b></div>
          </article>

          <article className="capacity-card">
            <p className="micro">EXECUTION CAPACITY</p>
            <h3>Scaling turns one trade into many buy-limit windows.</h3>
            <div className="capacity-value"><strong>{capacityPoint.capital}</strong><span>gp starting capital</span></div>
            <input
              aria-label="Starting capital scenario"
              type="range"
              min="0"
              max={capacity.length - 1}
              step="1"
              value={capacityIndex}
              onChange={(event) => setCapacityIndex(Number(event.target.value))}
            />
            <div className="range-labels"><span>728k</span><span>100m</span></div>
            <div className="capacity-stats">
              <div><span>Multi-window entries</span><strong>{capacityPoint.windows}</strong></div>
              <div><span>Worst windows needed</span><strong>{capacityPoint.max}</strong></div>
            </div>
            <p className={`capacity-verdict ${capacityIndex > 1 ? "danger" : capacityIndex > 0 ? "warn" : ""}`}><span /> {capacityPoint.verdict}</p>
          </article>
        </div>

        <div className="figure-card wide-figure">
          <Image width={1428} height={858} src="/research/cost-comparison.png" alt="Log-scale growth comparison of the tax-only and realistic-cost hide strategies from 2017 through 2026" />
          <p className="figure-caption"><span>Figure 2</span> Growth of one unit of capital. The separation is the cost of crossing actual market spreads.</p>
        </div>
      </section>

      <section className="section contrast" id="comparison">
        <div className="shell">
          <div className="section-heading">
            <div><p className="kicker">04 · Model comparison</p><h2>The OU baseline outperformed the ML alternatives.</h2></div>
            <p>Logistic regression and gradient-boosted trees use lagged features and the same walk-forward discipline. Neither beats the interpretable OU baseline.</p>
          </div>
          <div className="model-table" role="table" aria-label="Model performance comparison">
            <div className="model-row model-head" role="row"><span>Signal</span><span>Annual return</span><span>Sharpe</span><span>Max DD</span><span>Rotations</span><span>DSR</span></div>
            {mlModels.map((model, index) => (
              <div className={`model-row ${index === 0 ? "winner" : ""}`} role="row" key={model.name}>
                <span><b>{model.name}</b>{index === 0 && <small>Baseline</small>}</span>
                <span className={model.annual < 0 ? "negative" : ""}><i style={{ width: `${Math.max(3, Math.abs(model.annual))}%` }} />{model.annual.toFixed(1)}%</span>
                <span>{model.sharpe.toFixed(2)}</span><span>{model.dd}</span><span>{model.rotations}</span><span>{model.dsr}</span>
              </div>
            ))}
          </div>
          <p className="table-note">Comparison run uses a shared date range, giving the OU baseline 39.1% annualised return and 25 rotations; the longer primary backtest reports 35.1% and 34 rotations.</p>
        </div>
      </section>

      <section className="section shell" id="method">
        <div className="section-heading">
          <div><p className="kicker">05 · Methodology</p><h2>Walk-forward estimation limits lookahead.</h2></div>
          <p>The OU process is refit every 30 days on the trailing 730 days. Parameters are frozen until the next refit, making the signal genuinely out-of-sample.</p>
        </div>
        <div className="method-grid">
          <article className="timeline-card">
            <div className="timeline">
              <div><span>01</span><strong>Observe</strong><p>730 trailing daily closes</p></div>
              <div><span>02</span><strong>Fit</strong><p>β, mean, equilibrium σ, half-life</p></div>
              <div><span>03</span><strong>Freeze</strong><p>Hold parameters for 30 days</p></div>
              <div><span>04</span><strong>Trade</strong><p>Rotate only after threshold crossings</p></div>
            </div>
            <div className="audit-band"><span className="check">✓</span><p><strong>Lookahead audit passed</strong>0 structural violations across 3,328 hide rows; 8/8 sampled refits matched an independent recomputation.</p></div>
          </article>
          <article className="guardrails">
            <p className="micro">WHAT THE RESULT DOES NOT CLAIM</p>
            <ul>
              <li><span>01</span><p><strong>Not classical cointegration proof</strong>Both hide price legs are I(0), so the result is framed as stable mean reversion.</p></li>
              <li><span>02</span><p><strong>Not a frictionless P&amp;L</strong>Spread, tax, and hard Grand Exchange buy limits are explicit.</p></li>
              <li><span>03</span><p><strong>Not production execution</strong>Daily closes omit intraday queue position, fill uncertainty, and game-update risk.</p></li>
              <li><span>04</span><p><strong>Not threshold optimisation</strong>Fixed rules prevent this dashboard becoming an in-sample strategy tuner.</p></li>
            </ul>
          </article>
        </div>
      </section>

      <section className="closing shell">
        <p className="kicker">Project summary</p>
        <h2>A reproducible research pipeline, not a production trading system.</h2>
        <div>
          <p>GEcko covers data provenance, statistical diagnostics, walk-forward modelling, execution-aware backtesting, multiple-testing correction, ML benchmarking and automated lookahead audits.</p>
          <a className="button primary" href="https://github.com/TomGreenwoodPhysics/GEcko" target="_blank" rel="noreferrer">View the repository <Arrow /></a>
        </div>
      </section>

      <footer className="footer shell">
        <a className="brand" href="#top"><span>GEcko</span></a>
        <p>Independent research project · Not financial advice · Old School RuneScape is a trademark of Jagex Ltd.</p>
        <a href="#top">Back to top ↑</a>
      </footer>
    </main>
  );
}
