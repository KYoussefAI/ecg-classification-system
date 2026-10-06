import { useEffect, useRef } from "react";
import { LEADS } from "../utils/synthetic";

const ARRANGEMENT = [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11];

function Trace({ signal, lead, height = 125 }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !signal) return;
    const draw = () => {
      const width = canvas.clientWidth;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      const ctx = canvas.getContext("2d");
      ctx.scale(dpr, dpr);
      const left = 38,
        right = width - 12,
        top = 27,
        bottom = height - 24;
      const values = signal.map((row) => row[lead]);
      const range = Math.max(0.5, ...values.map(Math.abs)) * 1.15;
      const mid = (top + bottom) / 2;
      const y = (value) => mid - ((value / range) * (bottom - top)) / 2;
      ctx.strokeStyle = "#1b3039";
      ctx.lineWidth = 0.5;
      for (let second = 0; second <= 10; second += 0.5) {
        const x = left + ((right - left) * second) / 10;
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, bottom);
        ctx.stroke();
      }
      for (let step = -2; step <= 2; step++) {
        const py = y((range * step) / 2);
        ctx.beginPath();
        ctx.moveTo(left, py);
        ctx.lineTo(right, py);
        ctx.stroke();
      }
      ctx.strokeStyle = "#62d6d0";
      ctx.lineWidth = 1.15;
      ctx.lineJoin = "round";
      ctx.beginPath();
      values.forEach((value, index) => {
        const x = left + ((right - left) * (index / 100)) / 10;
        if (index) ctx.lineTo(x, y(value));
        else ctx.moveTo(x, y(value));
      });
      ctx.stroke();
      ctx.font = "10px ui-monospace, monospace";
      ctx.fillStyle = "#8499a6";
      ctx.fillText(`${range.toFixed(1)}`, 4, top + 4);
      ctx.fillText("0", 18, mid + 3);
      ctx.fillText(`−${range.toFixed(1)}`, 0, bottom + 3);
      ctx.fillText("mV", 5, 15);
      for (const second of [0, 5, 10])
        ctx.fillText(
          `${second}s`,
          left + ((right - left) * second) / 10 - (second === 10 ? 16 : 0),
          height - 6,
        );
    };
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    draw();
    return () => observer.disconnect();
  }, [signal, lead, height]);
  return (
    <div className="trace">
      <span className="trace-label">{LEADS[lead]}</span>
      <canvas
        ref={ref}
        style={{ height }}
        role="img"
        aria-label={`Lead ${LEADS[lead]}, 10 second waveform; automatically scaled amplitude in millivolts`}
      />
    </div>
  );
}

export default function Waveform({
  signal,
  rhythm = true,
  compact = false,
  analyzing = false,
}) {
  return (
    <div className={`waveform ${analyzing ? "analyzing" : ""}`}>
      {compact ? (
        <Trace signal={signal} lead={1} height={210} />
      ) : (
        <>
          <div className="lead-grid">
            {ARRANGEMENT.map((lead) => (
              <Trace key={lead} signal={signal} lead={lead} />
            ))}
          </div>
          {rhythm && (
            <div className="rhythm">
              <div className="eyebrow">Lead II · continuous rhythm strip</div>
              <Trace signal={signal} lead={1} height={135} />
            </div>
          )}
        </>
      )}
      <div className="waveform-caption">
        <span>Time: 0–10 s · Amplitude: mV</span>
        <span>Per-lead auto scale · not a calibrated ECG printout</span>
      </div>
    </div>
  );
}
