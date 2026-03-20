import type { ChatMessage } from "@/types/graph";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-3`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
          isUser
            ? "bg-secondary text-surface-0 rounded-br-md"
            : "bg-surface-50 text-surface-800 border border-surface-200 rounded-bl-md"
        }`}
      >
        {message.content}
      </div>
    </div>
  );
}
