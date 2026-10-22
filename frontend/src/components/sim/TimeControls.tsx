"use client";

import { Pause, Play } from "lucide-react";

// Days per real second. 365× plays a year per second.
const SPEEDS = [1, 5, 25, 100, 365];

interface TimeControlsProps {
  playing: boolean;
  speed: number;
  tick: number;
  maxTicks: number;
  isComplete: boolean;
  onPlay: () => void;
  onPause: () => void;
  onSetSpeed: (speed: number) => void;
}

export function TimeControls({
  playing,
  speed,
  tick,
  maxTicks,
  isComplete,
  onPlay,
  onPause,
  onSetSpeed,
}: TimeControlsProps) {
  const progress = maxTicks > 0 ? Math.min(tick / maxTicks, 1) : 0;
  const year = Math.max(1, Math.ceil(tick / 365));

  return (
    <div className="flex items-center gap-2 bg-surface-50/80 backdrop-blur border border-surface-200 rounded-xl px-3 py-1.5">
      <button
        onClick={playing ? onPause : onPlay}
        disabled={isComplete}
        className="flex items-center justify-center w-7 h-7 rounded-lg bg-surface-100 hover:bg-surface-200 disabled:opacity-40 text-surface-900 transition-all active:scale-[0.97]"
        data-tooltip={playing ? "Pause" : "Play"}
        data-tooltip-pos="below"
      >
        {playing ? (
          <Pause size={12} fill="currentColor" strokeWidth={0} />
        ) : (
          <Play size={12} fill="currentColor" strokeWidth={0} />
        )}
      </button>

      <div className="h-4 w-px bg-surface-200" />

      <div className="flex items-center gap-0.5">
        {SPEEDS.map((s) => (
          <button
            key={s}
            onClick={() => onSetSpeed(s)}
            data-tooltip={`${s} days / second`}
            data-tooltip-pos="below"
            className={`text-[10px] font-medium px-2 py-1 rounded-md transition-all active:scale-[0.97] ${
              speed === s
                ? "bg-accent text-surface-0 shadow-sm shadow-accent/30"
                : "text-surface-500 hover:text-surface-700 hover:bg-surface-100"
            }`}
          >
            {s}x
          </button>
        ))}
      </div>

      <div className="h-4 w-px bg-surface-200" />

      <div className="flex items-center gap-1.5 px-1">
        <div
          className={`w-1.5 h-1.5 rounded-full ${playing ? "bg-accent animate-pulse" : isComplete ? "bg-complete" : "bg-surface-400"}`}
        />
        <span className="text-xs text-surface-600">
          Year <span className="text-surface-800 font-mono font-medium">{year}</span> · Day{" "}
          <span className="text-surface-800 font-mono font-medium">{tick}</span>
        </span>
      </div>

      <div className="w-24 h-1.5 bg-surface-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-300 ${isComplete ? "bg-complete" : "bg-accent"}`}
          style={{ width: `${progress * 100}%` }}
        />
      </div>
      <span className="text-[10px] font-mono text-surface-500 w-8 text-right">
        {Math.round(progress * 100)}%
      </span>
    </div>
  );
}
