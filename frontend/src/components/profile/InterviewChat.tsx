"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Loader2, Paperclip } from "lucide-react";

interface Message {
  role: "assistant" | "user";
  content: string;
}

interface InterviewChatProps {
  messages: Message[];
  isLoading: boolean;
  isUploading?: boolean;
  progress: number;
  documents?: Array<{ filename: string; summary: string }>;
  onSendAnswer: (answer: string) => void;
  onUpload?: (file: File) => void;
}

export function InterviewChat({
  messages,
  isLoading,
  isUploading,
  progress,
  documents = [],
  onSendAnswer,
  onUpload,
}: InterviewChatProps) {
  const [input, setInput] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file && onUpload) {
        onUpload(file);
      }
      if (e.target) e.target.value = "";
    },
    [onUpload],
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!isLoading) inputRef.current?.focus();
  }, [isLoading]);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      const trimmed = input.trim();
      if (!trimmed || isLoading) return;
      onSendAnswer(trimmed);
      setInput("");
    },
    [input, isLoading, onSendAnswer],
  );

  const questionCount = messages.filter((m) => m.role === "assistant").length;

  return (
    <div className="flex flex-col h-full">
      {/* Progress bar */}
      <div className="px-4 py-3 border-b border-surface-200/50">
        <div className="flex items-center justify-between mb-1.5">
          <span className="text-[11px] font-medium text-surface-500 uppercase tracking-wider">
            Interview Progress
          </span>
          <span className="text-[11px] text-surface-400">
            Question {questionCount} of ~10
          </span>
        </div>
        <div className="h-1.5 bg-surface-100 rounded-full overflow-hidden">
          <div
            className="h-full bg-accent rounded-full transition-all duration-500"
            style={{ width: `${Math.round(progress * 100)}%` }}
          />
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[85%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed ${
                msg.role === "user"
                  ? "bg-accent text-white rounded-br-md"
                  : "bg-surface-100 text-surface-800 rounded-bl-md"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-surface-100 text-surface-500 px-4 py-2.5 rounded-2xl rounded-bl-md">
              <Loader2 size={16} className="animate-spin" />
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Document badges */}
      {documents.length > 0 && (
        <div className="px-4 py-2 border-t border-surface-200/50 flex flex-wrap gap-1.5">
          {documents.map((doc, i) => (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-2 py-1 bg-surface-100 rounded-md text-[11px] text-surface-600"
            >
              <Paperclip size={10} />
              {doc.filename}
            </span>
          ))}
        </div>
      )}

      {/* Input */}
      <form
        onSubmit={handleSubmit}
        className="px-4 py-3 border-t border-surface-200/50"
      >
        <div className="flex items-center gap-2">
          {onUpload && (
            <>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.txt,.csv,.md"
                onChange={handleFileChange}
                className="hidden"
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading || isUploading}
                className="flex items-center justify-center w-10 h-10 rounded-xl border border-surface-200 text-surface-500 transition-all hover:border-surface-300 hover:text-surface-700 disabled:opacity-40 disabled:cursor-not-allowed"
                title="Upload a business document"
              >
                {isUploading ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Paperclip size={16} />
                )}
              </button>
            </>
          )}
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your answer..."
            disabled={isLoading}
            className="flex-1 bg-surface-50 border border-surface-200 rounded-xl px-4 py-2.5 text-sm text-surface-800 placeholder-surface-400 outline-none focus:border-accent focus:ring-1 focus:ring-accent/30 disabled:opacity-50"
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="flex items-center justify-center w-10 h-10 rounded-xl bg-accent text-white transition-all hover:bg-accent/90 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            <Send size={16} />
          </button>
        </div>
      </form>
    </div>
  );
}
