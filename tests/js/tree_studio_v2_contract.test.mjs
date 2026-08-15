// Behavioral (not string-matching) tests for project/tree_studio.html's node-form
// contract: lossless round-trip, explicit-zero preservation, and corrected
// backward/raw terminology. Run with `npm test` (node --test tests/js/).
//
// Loads the real page into jsdom and drives its actual renderNode()/
// applyNodeForm() functions - this is not a reimplementation of the app's logic,
// it exercises the live DOM the browser would build.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { JSDOM } from 'jsdom';

const here = path.dirname(fileURLToPath(import.meta.url));
const html = readFileSync(path.join(here, '..', '..', 'project', 'tree_studio.html'), 'utf-8');

// The page's top-level `let state = null;` is a global *lexical* binding, not
// a property of `window` - `window.state = x` would silently not affect what
// the app's own functions see. Setting it through an indirect eval (and also
// mirroring it onto window.state so the test can read mutations back) shares
// the same global lexical environment the script tag itself declared into.
function setAppState(window, config, selectedId) {
  window.eval('state = ' + JSON.stringify(config) + '; window.state = state; ' +
    'selectedId = ' + JSON.stringify(selectedId) + ';');
}

function makeWindow() {
  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    url: 'http://localhost/',
    pretendToBeVisual: true,
    beforeParse(window) {
      // The app's own bootstrap does `fetch('/api/sample').then(r=>r.json())
      // .then(data=>{state=data;...render();})` with no .catch(). A never-
      // settling promise keeps that chain from ever running (and later
      // clobbering the state we set for the test, or crashing render() on an
      // empty stub payload) while still giving `fetch` a defined global so
      // the line doesn't throw synchronously while the script loads.
      window.fetch = () => new Promise(() => {});
      const store = new Map();
      window.localStorage = {
        getItem: (key) => (store.has(key) ? store.get(key) : null),
        setItem: (key, value) => store.set(key, String(value)),
        removeItem: (key) => store.delete(key),
      };
    },
  });
  return dom.window;
}

function fullConfig() {
  return {
    root_id: 'root',
    // renderGlobal() reads state.data directly; a full render() (triggered by
    // switching objective, see the HRP test below) throws without it.
    data: { start: '2020-01-01', end: '2025-01-01', risk_free_annual: 0, borrow_spread_bps: 0 },
    nodes: [
      {
        id: 'root',
        name: 'Root',
        children: [],
        instruments: ['ACWI', 'AGG'],
        representation: 'synthetic',
        proxy: '',
        inherits: 'none',
        goal: { objective: 'min_risk' },
        constraints: {
          min_weights: {},
          max_weights: {},
          per_asset_cap: '',
          max_turnover: '',
          cash_enabled: false,
          max_leverage: '1',
          borrow_spread_bps: '',
          volatility_reference: 'none',
          vol_target: '',
          volatility_target_policy: 'nearest_feasible',
          max_volatility_reference: 'none',
          max_volatility: '',
          max_tracking_error: '',
          tracking_error_reference: 'declared',
          tracking_error_policy: 'hard_fail',
          volatility_cap_policy: 'hard_fail',
          mean_estimator: 'auto',
          mean_reference_kind: 'local_weights',
          mean_reference_weights: { ACWI: 0.7, AGG: 0.3 },
          risk_aversion: '',
          risk_free_rate: 0,
          // Fields with no GUI control at all - must survive purely via merge.
          views: [{ instruments: { ACWI: 1.0 }, expected_return: 0.05, confidence: 0.5, source: 'llm' }],
          view_tau: 0.07,
          covariance_estimator: 'ledoit_wolf',
          view_covariance_policy: 'posterior_all',
          some_unknown_future_field: { nested: true, value: 42 },
        },
        benchmarks: [],
      },
    ],
    backtest: { benchmark: { weights: { ACWI: 0.7, AGG: 0.3 } } },
  };
}

test('explicit zero risk_free_rate renders as "0", not blank', () => {
  const window = makeWindow();
  setAppState(window, fullConfig(), 'root');
  window.renderNode();
  const field = window.document.getElementById('c-risk-free-rate');
  assert.equal(field.value, '0', 'an explicit 0 must render as the digit 0, not an empty/blank input');
});

test('editing one unrelated field preserves zero, views, and unknown fields (lossless round trip)', () => {
  const window = makeWindow();
  setAppState(window, fullConfig(), 'root');
  window.renderNode();

  // Sanity: the field we are about to "edit" starts empty (risk_aversion).
  assert.equal(window.document.getElementById('c-risk-aversion').value, '');

  // Simulate the user touching exactly one, unrelated field.
  window.document.getElementById('c-risk-aversion').value = '2.5';
  window.applyNodeForm();

  const constraints = window.state.nodes[0].constraints;

  // The edited field took effect.
  assert.equal(constraints.risk_aversion, '2.5');

  // Explicit zero for risk_free_rate must not become '', null, or inherited.
  assert.equal(Number(constraints.risk_free_rate), 0);
  assert.notEqual(constraints.risk_free_rate, '');

  // Views and view_tau now have real GUI controls (see the dedicated Views
  // tests below), so - like every other constraint field with a control -
  // they round-trip through DOM input values as strings, not their original
  // JS types. covariance_estimator/view_covariance_policy/the unknown field
  // below still have no control at all and must pass through untouched.
  // (JSON round-trip sidesteps a cross-realm prototype mismatch between this
  // Node.js process's Object/Array and the jsdom window's own globals - the
  // objects are structurally identical, just from a different realm.)
  assert.equal(
    JSON.stringify(constraints.views),
    JSON.stringify([
      { instruments: { ACWI: '1' }, expected_return: '0.05', confidence: '0.5', source: 'llm' },
    ])
  );
  assert.equal(constraints.view_tau, '0.07');
  assert.equal(constraints.covariance_estimator, 'ledoit_wolf');
  assert.equal(constraints.view_covariance_policy, 'posterior_all');
  assert.equal(
    JSON.stringify(constraints.some_unknown_future_field),
    JSON.stringify({ nested: true, value: 42 })
  );

  // The new mean-reference axis (has a form control) is still present and
  // independent of the risk-side volatility_reference, which was untouched.
  assert.equal(constraints.mean_reference_kind, 'local_weights');
  assert.equal(Number(constraints.mean_reference_weights.ACWI), 0.7);
  assert.equal(Number(constraints.mean_reference_weights.AGG), 0.3);
  assert.equal(constraints.volatility_reference, 'none');
});

test('the "forward_root_reference" option replaces the old ambiguous "root" value', () => {
  const window = makeWindow();
  setAppState(window, fullConfig(), 'root');
  window.renderNode();
  const select = window.document.getElementById('c-vol-reference');
  const values = [...select.options].map((option) => option.value);
  assert.ok(values.includes('forward_root_reference'));
  assert.ok(!values.includes('root'));
});

test('constraint-policy controls are present and round-trip', () => {
  const window = makeWindow();
  setAppState(window, fullConfig(), 'root');
  window.renderNode();
  assert.equal(window.document.getElementById('c-vol-target-policy').value, 'nearest_feasible');
  assert.equal(window.document.getElementById('c-te-policy').value, 'hard_fail');

  window.document.getElementById('c-vol-target-policy').value = 'hard_fail';
  window.applyNodeForm();
  assert.equal(window.state.nodes[0].constraints.volatility_target_policy, 'hard_fail');
});

function equityConfig() {
  const config = fullConfig();
  config.nodes[0].instruments = ['SPY', 'VGK', 'EWJ'];
  config.nodes[0].goal = { objective: 'max_return' };
  config.nodes[0].constraints.views = [];
  return config;
}

test('views section is structurally separate from the constraints section', () => {
  const window = makeWindow();
  setAppState(window, equityConfig(), 'root');
  window.renderNode();
  const viewsSection = window.document.getElementById('viewsSection');
  assert.ok(viewsSection, 'expected a dedicated #viewsSection element');
  // It must not be nested inside a "Constraints" section, and vice versa.
  const constraintsHeading = [...window.document.querySelectorAll('.section h2')]
    .find((h2) => /Constraints/i.test(h2.textContent));
  const constraintsSection = constraintsHeading.closest('.section');
  assert.ok(!constraintsSection.contains(viewsSection), 'views must not be nested inside Constraints');
  assert.ok(!viewsSection.contains(constraintsSection), 'constraints must not be nested inside views');
});

test('a relative view (signed, multi-instrument) round-trips through the ticker picker', () => {
  const window = makeWindow();
  setAppState(window, equityConfig(), 'root');
  window.renderNode();

  window.document.getElementById('addView').click();
  const spyBox = window.document.querySelector('[data-view-ticker="SPY"]');
  const vgkBox = window.document.querySelector('[data-view-ticker="VGK"]');
  spyBox.checked = true;
  vgkBox.checked = true;
  window.document.querySelector('[data-view-weight="SPY"]').value = '1';
  window.document.querySelector('[data-view-weight="VGK"]').value = '-1';
  window.document.querySelector('[data-view-return]').value = '0.05';
  window.document.querySelector('[data-view-confidence]').value = '0.6';
  window.applyNodeForm();

  const [view] = window.state.nodes[0].constraints.views;
  assert.equal(view.instruments.SPY, '1');
  assert.equal(view.instruments.VGK, '-1');
  assert.equal(Object.keys(view.instruments).length, 2, 'EWJ was never checked, must not appear');
  assert.equal(view.expected_return, '0.05');
  assert.equal(view.confidence, '0.6');
});

test('unchecking a ticker clears its paired weight input', () => {
  const window = makeWindow();
  setAppState(window, equityConfig(), 'root');
  window.renderNode();

  window.document.getElementById('addView').click();
  const spyBox = window.document.querySelector('[data-view-ticker="SPY"]');
  const weight = window.document.querySelector('[data-view-weight="SPY"]');
  spyBox.checked = true;
  spyBox.dispatchEvent(new window.Event('change'));
  weight.value = '1';
  assert.equal(weight.disabled, false);

  spyBox.checked = false;
  spyBox.dispatchEvent(new window.Event('change'));
  assert.equal(weight.disabled, true);
  assert.equal(weight.value, '', 'weight must be cleared, not just disabled, when its ticker is unchecked');
});

test('a fully-empty view row (added but never filled in) is dropped, not sent to the backend', () => {
  const window = makeWindow();
  setAppState(window, equityConfig(), 'root');
  window.renderNode();

  window.document.getElementById('addView').click();
  window.document.getElementById('addView').click();
  const spyBox = window.document.querySelectorAll('[data-view-ticker="SPY"]')[0];
  spyBox.checked = true;
  window.document.querySelectorAll('[data-view-weight="SPY"]')[0].value = '1';
  window.document.querySelectorAll('[data-view-return]')[0].value = '0.05';
  window.document.querySelectorAll('[data-view-confidence]')[0].value = '0.6';
  window.applyNodeForm();

  assert.equal(window.state.nodes[0].constraints.views.length, 1, 'the second, untouched row must be dropped');
});

test('view_tau round-trips through its own field', () => {
  const window = makeWindow();
  const config = equityConfig();
  config.nodes[0].constraints.view_tau = 0.12;
  setAppState(window, config, 'root');
  window.renderNode();
  assert.equal(window.document.getElementById('c-view-tau').value, '0.12');
  window.document.getElementById('c-view-tau').value = '0.2';
  window.applyNodeForm();
  assert.equal(window.state.nodes[0].constraints.view_tau, '0.2');
});

test('switching objective to HRP destructively clears views and rebuilds the DOM; switching back does not resurrect them', () => {
  const window = makeWindow();
  const config = equityConfig();
  config.nodes[0].constraints.views = [
    { instruments: { SPY: 1 }, expected_return: 0.05, confidence: 0.5, source: 'manual' },
  ];
  setAppState(window, config, 'root');
  window.renderNode();
  assert.equal(window.document.querySelectorAll('.view-item').length, 1);

  const objective = window.document.getElementById('n-objective');
  objective.value = 'hrp';
  objective.dispatchEvent(new window.Event('change'));
  assert.equal(window.state.nodes[0].constraints.views.length, 0, 'HRP must clear views in state');
  assert.equal(
    window.document.querySelectorAll('.view-item').length,
    0,
    'the DOM must be rebuilt too, not just the state - a stale row would resurrect on the next form read'
  );

  objective.value = 'max_return';
  objective.dispatchEvent(new window.Event('change'));
  assert.equal(
    window.state.nodes[0].constraints.views.length,
    0,
    'a cleared view must not come back just from toggling the objective away from HRP and back'
  );
});

test('mean-reference terminology tooltip states father/B0 stay raw, not synthetic, in backward', () => {
  const window = makeWindow();
  setAppState(window, fullConfig(), 'root');
  window.renderNode();
  const icon = window.document
    .getElementById('c-vol-reference')
    .closest('.field')
    .querySelector('label .info');
  assert.ok(icon, 'expected a help icon attached to the volatility-reference label');
  const tip = icon.getAttribute('data-tip');
  assert.ok(
    !/Father diventa la serie sintetica/i.test(tip),
    'tooltip must not claim father becomes synthetic in the backward pass'
  );
  assert.ok(
    /SEMPRE/.test(tip) && /raw/i.test(tip),
    'tooltip must state father/B0 always stay raw'
  );
});

test('adaptive pruning controls round-trip as backend policy parameters', () => {
  const window = makeWindow();
  const config = fullConfig();
  setAppState(window, config, 'root');
  window.renderGlobal();

  window.document.getElementById('prune-enabled').checked = true;
  window.document.getElementById('prune-burn-in').value = '1.5';
  window.document.getElementById('prune-window').value = '3';
  window.document.getElementById('prune-sharpe').value = '0.05';
  window.document.getElementById('prune-drawdown').value = '1.2';
  window.document.getElementById('prune-workers').value = '6';
  window.document.getElementById('prune-max-folds').value = '24';
  window.document.getElementById('prune-expanding').checked = true;
  window.applyGlobal();

  assert.deepEqual(JSON.parse(JSON.stringify(window.state.backtest.adaptive_pruning)), {
    enabled: true,
    burn_in_years: 1.5,
    evidence_window_years: 3,
    min_sharpe_improvement: 0.05,
    max_drawdown_per_vol_ratio: 1.2,
    workers: 6,
    max_folds: 24,
    expanding: true,
  });
});
