"use client";

import { useCallback, useState } from "react";
import { ModelPicker } from "@/components/onboarding/ModelPicker";
import { SimSetup } from "@/components/onboarding/SimSetup";
import { SimView } from "@/components/sim/SimView";
import { useSimulation } from "@/hooks/useSimulation";
import type { AppConfig, ModelOption, StartRequest } from "@/lib/api";

type Screen = "model" | "setup" | "sim";

export default function Home() {
  const sim = useSimulation();
  const [screen, setScreen] = useState<Screen>("model");
  const [model, setModel] = useState<ModelOption | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [aiRun, setAiRun] = useState(false);

  const start = useCallback(
    (req: StartRequest) => {
      setAiRun(req.ai_enabled);
      setScreen("sim");
      void sim.start(req);
    },
    [sim],
  );

  const backToSetup = useCallback(() => {
    void sim.stop();
    setScreen("setup");
  }, [sim]);

  if (screen === "model" || !model || !config) {
    return (
      <ModelPicker
        onContinue={(m, cfg) => {
          setModel(m);
          setConfig(cfg);
          setScreen("setup");
        }}
      />
    );
  }

  if (screen === "setup") {
    return (
      <SimSetup model={model} config={config} onBack={() => setScreen("model")} onStart={start} />
    );
  }

  return <SimView sim={sim} modelName={aiRun ? model.name : null} onBack={backToSetup} />;
}
