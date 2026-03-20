"use client";

import { useState, useRef, useEffect } from "react";
import type { ChatMessage, CompanyGraph } from "@/types/graph";
import { parseCompany, refineCompany } from "@/lib/api";
import { MessageBubble } from "./MessageBubble";
import { MessageSquare } from "@/components/ui/icons";

interface ChatPanelProps {
  graph: CompanyGraph | null;
  onGraphUpdate: (graph: CompanyGraph) => void;
}

export function ChatPanel({ graph, onGraphUpdate }: ChatPanelProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = input.trim();
    if (!trimmed || loading) return;

    const userMsg: ChatMessage = { role: "user", content: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      let result: CompanyGraph;
      if (graph) {
        result = await refineCompany(trimmed, graph);
      } else {
        result = await parseCompany(trimmed);
      }
      onGraphUpdate(result);
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: `Updated "${result.name}" — ${result.nodes.length} nodes, ${result.edges.length} connections.`,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      const errorMsg: ChatMessage = {
        role: "assistant",
        content: `Error: ${err instanceof Error ? err.message : "Something went wrong"}`,
      };
      setMessages((prev) => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-surface-0">
      {/* Header */}
      <div className="px-5 py-4 border-b border-surface-200/50">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-secondary" />
          <h2 className="text-sm font-semibold text-surface-800">Company Builder</h2>
        </div>
        <p className="text-xs text-surface-500 mt-1 ml-4">
          {graph ? `Editing "${graph.name}"` : "Describe your company to generate a structure"}
        </p>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4">
        {messages.length === 0 && (
          <div className="mt-12 text-center space-y-4">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-2xl bg-surface-50 border border-surface-200">
              <MessageSquare size={20} className="text-surface-400" />
            </div>
            <div>
              <p className="text-surface-500 text-sm">Try something like:</p>
              <div className="mt-3 space-y-2">
                {[
                  "A SaaS startup with engineering, sales, and marketing",
                  "An e-commerce company with 200 employees",
                  "A consulting firm with three practice areas",
                ].map((suggestion) => (
                  <button
                    key={suggestion}
                    onClick={() => setInput(suggestion)}
                    className="block w-full text-left text-xs text-surface-600 hover:text-surface-800 bg-surface-50/50 hover:bg-surface-50 border border-surface-200/50 hover:border-surface-300 rounded-xl px-4 py-2.5 transition-all"
                  >
                    &ldquo;{suggestion}&rdquo;
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
        {messages.map((msg, i) => (
          <MessageBubble key={i} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start mb-3">
            <div className="bg-surface-50 border border-surface-200 rounded-2xl rounded-bl-md px-4 py-3 text-sm text-surface-600">
              <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 rounded-full bg-surface-400 animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Input */}
      <div className="p-4 border-t border-surface-200/50">
        <form onSubmit={handleSubmit} className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={graph ? "Refine your company..." : "Describe a company..."}
            disabled={loading}
            className="w-full bg-surface-50 border border-surface-200 hover:border-surface-300 rounded-xl px-4 py-3 pr-20 text-sm text-surface-900 placeholder-surface-500 focus:outline-none focus:ring-2 focus:ring-secondary/30 focus:border-secondary/30 disabled:opacity-50 transition-all"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="absolute right-2 top-1/2 -translate-y-1/2 bg-secondary hover:bg-secondary-dim disabled:bg-surface-200 disabled:text-surface-500 text-surface-0 text-xs font-medium px-3 py-1.5 rounded-lg transition-all active:scale-[0.97]"
          >
            {loading ? "..." : "Send"}
          </button>
        </form>
      </div>
    </div>
  );
}
