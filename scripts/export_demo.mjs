import { mkdirSync, writeFileSync } from 'node:fs'
import { syntheticSignal, LEADS } from '../frontend/src/utils/synthetic.js'

mkdirSync('artifacts/demo', { recursive: true })
writeFileSync('artifacts/demo/synthetic_demo.json', JSON.stringify({
  signal_data: syntheticSignal(), leads: LEADS, sample_rate: 100, units: 'mV',
  source: 'Synthetic demo signal — illustrative, not patient data',
  synthetic: true,
}))
console.log('Wrote artifacts/demo/synthetic_demo.json (synthetic; no diagnostic ground truth).')
