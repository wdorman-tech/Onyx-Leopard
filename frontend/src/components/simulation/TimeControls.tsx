"use client";

import { Play, Pause } from "@/components/ui/icons";

interface TimeControlsProps {
  playing: boolean;
  speed: number;
  tick: number;
  isComplete: boolean;
  onPlay: () => void;
  onPause: () => void;
  onSetSpeed: (speed: number) => void;
}

const SPEEDS = [1, 2, 5, 10];

export function TimeControls({
  playing,
  speed,
  tick,
  isComplete,
  onPlay,
  onPause,
  onSetSpeed,
}: TimeControlsProps) {
  return (
    <div className="flex items-center gap-2 bg-surface-50/80 backdrop-blur border border-surface-200 rounded-xl px-3 py-1.5">
      <button
        onClick={playing ? onPause : onPlay}
        disabled={isComplete}
        className="flex items-center justify-center w-7 h-7 rounded-lg bg-surface-100 hover:bg-surface-200 disabled:opacity-40 text-surface-900 transition-all active:scale-[0.97]"
        data-tooltip={playing ? "Pause" : "Play"}
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
            data-tooltip={`${s}x speed`}
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
        <div className={`w-1.5 h-1.5 rounded-full ${playing ? "bg-accent animate-pulse" : isComplete ? "bg-complete" : "bg-surface-400"}`} />
        <span className="text-xs text-surface-600">
          Week <span className="text-surface-800 font-mono font-medium">{tick}</span>
        </span>
      </div>

      {isComplete && (
        <span className="text-[10px] font-medium text-complete bg-complete/10 px-2 py-0.5 rounded-md">
          Done
        </span>
      )}
    </div>
  );
}
