"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { ArrowLeft, Loader2 } from "lucide-react";
import { Logo } from "@/components/ui/Logo";
import { InterviewChat } from "@/components/profile/InterviewChat";
import { ProfileSummary } from "@/components/profile/ProfileSummary";
import { useProfileBuilder } from "@/hooks/useProfileBuilder";

export default function ProfilePage() {
  const router = useRouter();
  const profile = useProfileBuilder();

  useEffect(() => {
    if (!profile.sessionId) {
      profile.start();
    }
  }, []);

  // After confirming, redirect back to home
  useEffect(() => {
    if (profile.confirmedSlug) {
      router.push("/");
    }
  }, [profile.confirmedSlug, router]);

  return (
    <div className="h-screen flex flex-col bg-surface-0">
      {/* Header */}
      <header className="flex items-center gap-3 px-5 py-3 border-b border-surface-200/50">
        <button
          onClick={() => router.push("/")}
          className="flex items-center justify-center w-8 h-8 rounded-lg hover:bg-surface-100 transition-colors"
        >
          <ArrowLeft size={16} className="text-surface-500" />
        </button>
        <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-surface-100 to-surface-50 border border-surface-200/50">
          <Logo size={16} className="text-accent" />
        </div>
        <div>
          <h1 className="text-sm font-bold text-surface-900 tracking-wide">
            Create Your Business
          </h1>
          <p className="text-[10px] text-surface-500 uppercase tracking-widest">
            AI Interview
          </p>
        </div>
      </header>

      {/* Content */}
      <div className="flex-1 overflow-hidden">
        {!profile.sessionId ? (
          <div className="flex items-center justify-center h-full">
            <Loader2 size={24} className="animate-spin text-surface-400" />
          </div>
        ) : profile.isComplete && profile.industrySpec ? (
          <ProfileSummary
            spec={profile.industrySpec}
            isLoading={profile.isLoading}
            error={profile.error}
            onConfirm={profile.confirm}
          />
        ) : (
          <InterviewChat
            messages={profile.messages}
            isLoading={profile.isLoading}
            isUploading={profile.isUploading}
            progress={profile.progress}
            documents={profile.documents}
            onSendAnswer={profile.sendAnswer}
            onUpload={profile.upload}
          />
        )}
      </div>
    </div>
  );
}
