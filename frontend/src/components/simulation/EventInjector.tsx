"use client";

import { useState } from "react";
import { Zap } from "@/components/ui/icons";

interface EventInjectorProps {
  onInject: (description: string) => void;
  disabled: boolean;
}

export function EventInjector({ onInject, disabled }: EventInjectorProps) {
  const [input, setInput] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;
    onInject(input.trim());
    setInput("");
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2">
      <div className="relative flex-1">
        <div className="absolute left-3 top-1/2 -translate-y-1/2 text-warning/60">
          <Zap size={14} />
        </div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Inject event: e.g. 'Major competitor launches rival product'"
          disabled={disabled}
          data-tooltip="Inject a market event into the simulation"
          className="w-full bg-surface-50/50 border border-surface-200/50 hover:border-surface-300 rounded-xl pl-9 pr-4 py-2 text-xs text-surface-900 placeholder-surface-500 focus:outline-none focus:ring-2 focus:ring-warning/20 focus:border-warning/30 disabled:opacity-40 transition-all"
        />
      </div>
      <button
        type="submit"
        disabled={disabled || !input.trim()}
        className="bg-warning/80 hover:bg-warning disabled:bg-surface-200 disabled:text-surface-500 text-surface-0 text-xs font-medium px-4 py-2 rounded-xl transition-all shadow-sm active:scale-[0.97]"
      >
        Inject
      </button>
    </form>
  );
}
