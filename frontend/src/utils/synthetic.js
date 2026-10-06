// Deterministic illustrative morphology. Not a patient recording or simulator
// validated for diagnosis. Limb signals obey the standard linear relationships.
export const LEADS = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
];
export function syntheticSignal() {
  const pulse = (t, r, s, p, tw) => {
    let y = 0;
    for (let beat = -1; beat < 14; beat++) {
      const d = t - (0.38 + beat * 0.81);
      const g = (at, width, amplitude) =>
        amplitude * Math.exp(-0.5 * ((d - at) / width) ** 2);
      y +=
        g(-0.17, 0.035, p) +
        g(-0.032, 0.009, -0.09) +
        g(0, 0.012, r) +
        g(0.035, 0.014, -s) +
        g(0.23, 0.065, tw);
    }
    return y;
  };
  return Array.from({ length: 1000 }, (_, i) => {
    const t = i / 100;
    const first = pulse(t, 0.8, 0.18, 0.09, 0.22);
    const second = pulse(t, 1.1, 0.24, 0.13, 0.3);
    const limb = [
      first,
      second,
      second - first,
      -(first + second) / 2,
      first - second / 2,
      second - first / 2,
    ];
    const chest = Array.from({ length: 6 }, (_, l) =>
      pulse(
        t,
        [0.18, 0.4, 0.8, 1.2, 1.05, 0.85][l],
        [0.7, 0.65, 0.5, 0.3, 0.15, 0.1][l],
        0.06,
        [0.08, 0.12, 0.22, 0.3, 0.26, 0.23][l],
      ),
    );
    return [...limb, ...chest].map(
      (v, l) => v + 0.007 * Math.sin(t * 2 * Math.PI * 0.25 + l * 0.15),
    );
  });
}
