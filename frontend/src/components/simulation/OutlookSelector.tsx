"use client";

interface OutlookSelectorProps {
  outlook: string;
  onSelect: (outlook: string) => void;
  disabled?: boolean;
}

const OUTLOOKS = [
  {
    value: "pessimistic",
    label: "Pessimistic",
    icon: "\u{1F4C9}",
    color: "negative",
    description: "Market downturn, tight budgets",
  },
  {
    value: "normal",
    label: "Normal",
    icon: "\u{1F4CA}",
    color: "surface",
    description: "Stable market conditions",
  },
  {
    value: "optimistic",
    label: "Optimistic",
    icon: "\u{1F4C8}",
    color: "accent",
    description: "Market boom, high growth",
  },
] as const;

const COLOR_MAP: Record<string, { active: string; text: string; dot: string }> = {
  negative: {
    active: "bg-negative/10 border-negative/40",
    text: "text-negative",
    dot: "bg-negative",
  },
  surface: {
    active: "bg-surface-400/10 border-surface-400/40",
    text: "text-surface-700",
    dot: "bg-surface-600",
  },
  accent: {
    active: "bg-accent/10 border-accent/40",
    text: "text-accent",
    dot: "bg-accent",
  },
};

export function OutlookSelector({ outlook, onSelect, disabled }: OutlookSelectorProps) {
  return (
    <div className="flex items-center gap-1.5">
      {OUTLOOKS.map((o) => {
        const isActive = outlook === o.value;
        const colors = COLOR_MAP[o.color];
        return (
          <button
            key={o.value}
            onClick={() => onSelect(o.value)}
            disabled={disabled}
            title={o.description}
            data-tooltip={o.description}
            className={`flex items-center gap-1.5 text-xs font-medium px-3 py-1.5 rounded-lg border transition-all disabled:opacity-40 active:scale-[0.97] ${
              isActive
                ? `${colors.active} ${colors.text}`
                : "border-surface-200 text-surface-500 hover:text-surface-700 hover:border-surface-300 hover:bg-surface-50/50"
            }`}
          >
            <span className="text-sm">{o.icon}</span>
            <span>{o.label}</span>
            {isActive && (
              <div className={`w-1.5 h-1.5 rounded-full ${colors.dot} animate-pulse`} />
            )}
          </button>
        );
      })}
    </div>
  );
}
