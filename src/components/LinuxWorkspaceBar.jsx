import React, { useEffect } from "react";
import { motion, useReducedMotion } from "framer-motion";
import {
  Sun,
  Moon,
  Terminal,
  AlertCircle,
  Layers,
} from "lucide-react";
import { WORKSPACE_TABS } from "../constants/workspaceTabs";

export default function LinuxWorkspaceBar({
  activeTab,
  onSelectTab,
  lightMode,
  onToggleLightMode,
  onOpenLimitations,
}) {
  const shouldReduceMotion = useReducedMotion();

  useEffect(() => {
    const onKey = (e) => {
      const activeEl = document.activeElement;
      const isInput =
        activeEl?.tagName === "INPUT" ||
        activeEl?.tagName === "TEXTAREA" ||
        activeEl?.isContentEditable;
      if (isInput) return;

      // Alt+1 to Alt+7
      if (e.altKey && e.key >= "1" && e.key <= "7") {
        e.preventDefault();
        onSelectTab(Number(e.key) - 1);
      }
      // Bracket keys [ and ] for previous and next workspace
      else if (e.key === "[") {
        e.preventDefault();
        onSelectTab((activeTab - 1 + WORKSPACE_TABS.length) % WORKSPACE_TABS.length);
      } else if (e.key === "]") {
        e.preventDefault();
        onSelectTab((activeTab + 1) % WORKSPACE_TABS.length);
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeTab, onSelectTab]);

  return (
    <aside className="w-64 lg:w-72 h-screen border-r border-slate-800 bg-slate-950 flex flex-col justify-between shrink-0 sticky top-0 p-4 select-none z-40 overflow-y-auto">
      <div className="space-y-5">
        {/* Header with Prototype & Scope Notice */}
        <div className="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-indigo-400 shrink-0" />
            <h1 className="font-mono text-xs font-bold tracking-tight text-slate-100">
              ComplaintLens
            </h1>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">
              prototype • read-only
            </span>
            <span className="text-[10px] font-mono text-slate-500">
              Alt+1..7, [ / ]
            </span>
          </div>
          <p className="text-[10px] text-slate-400 font-sans leading-tight">
            English review corpora across 4 platforms. Not validated for production use.
          </p>
        </div>

        {/* Workspace Navigation Tablist */}
        <nav
          role="tablist"
          aria-label="Workspaces"
          className="space-y-1"
        >
          <div className="px-2 pb-1 text-[10px] font-mono uppercase tracking-wider text-slate-500 flex items-center justify-between">
            <span>Workspaces</span>
            <Layers className="w-3 h-3 text-slate-600" />
          </div>

          {WORKSPACE_TABS.map((t) => {
            const Icon = t.icon;
            const active = activeTab === t.id;
            return (
              <button
                key={t.code}
                role="tab"
                id={`tab-${t.code}`}
                aria-selected={active}
                aria-controls={`panel-${t.code}`}
                tabIndex={active ? 0 : -1}
                onClick={() => onSelectTab(t.id)}
                className={`relative w-full px-3 py-2.5 rounded-xl text-xs font-medium flex items-center gap-2.5 text-left transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${
                  active
                    ? "text-white"
                    : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
                }`}
              >
                {active && !shouldReduceMotion && (
                  <motion.div
                    layoutId="activeWorkspace"
                    transition={{ type: "spring", stiffness: 380, damping: 32 }}
                    className="absolute inset-0 rounded-xl bg-indigo-500/20 border border-indigo-400/40 shadow-[0_0_20px_rgba(99,102,241,0.18)]"
                  />
                )}
                {active && shouldReduceMotion && (
                  <div className="absolute inset-0 rounded-xl bg-indigo-500/20 border border-indigo-400/40" />
                )}
                <Icon className="relative z-10 w-4 h-4 text-indigo-400 shrink-0" />
                <span className="relative z-10 font-mono text-[11px] text-indigo-400 font-semibold">
                  [{t.code}]
                </span>
                <span className="relative z-10 truncate text-slate-200">
                  {t.label}
                </span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer controls: Section 9 Limitations & Theme */}
      <div className="pt-4 border-t border-slate-800 space-y-2 mt-4">
        <button
          onClick={onOpenLimitations}
          className="w-full text-xs font-mono px-3 py-2 rounded-xl border border-slate-800 bg-slate-900 hover:border-amber-400/40 text-slate-300 hover:text-white flex items-center justify-between transition-colors focus:outline-none focus:ring-2 focus:ring-amber-400"
        >
          <span className="flex items-center gap-2">
            <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
            Limitations Drawer
          </span>
          <span className="text-[10px] text-slate-500 font-bold">Sec 9</span>
        </button>

        <button
          onClick={onToggleLightMode}
          className="w-full text-xs font-mono px-3 py-2 rounded-xl border border-slate-800 bg-slate-900 hover:border-slate-700 text-slate-300 flex items-center justify-between transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500"
        >
          <span className="text-slate-400">Palette</span>
          <span className="flex items-center gap-1.5 text-indigo-400">
            {lightMode ? <Moon className="w-3.5 h-3.5" /> : <Sun className="w-3.5 h-3.5" />}
            {lightMode ? "Slate Light" : "Slate-950 Dark"}
          </span>
        </button>
      </div>
    </aside>
  );
}
