import React, { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import LinuxWorkspaceBar from "./components/LinuxWorkspaceBar";
import { WORKSPACE_TABS } from "./constants/workspaceTabs";
import LimitationsDrawer from "./components/LimitationsDrawer";

import Tab1Corpus from "./components/tabs/Tab1Corpus";
import Tab2ChampionDeberta from "./components/tabs/Tab2ChampionDeberta";
import Tab3BenchmarksArchive from "./components/tabs/Tab3BenchmarksArchive";
import Tab4LivePredictor from "./components/tabs/Tab4LivePredictor";
import Tab5CrossDomain from "./components/tabs/Tab5CrossDomain";
import Tab6AiAgent from "./components/tabs/Tab6AiAgent";
import Tab7ErrorDiagnostics from "./components/tabs/Tab7ErrorDiagnostics";

class WorkspaceErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error("Workspace render error:", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-8 rounded-2xl border border-rose-500/30 bg-rose-950/20 text-center font-mono space-y-3">
          <div className="text-rose-400 font-bold text-base">
            Workspace Render Notice
          </div>
          <div className="text-xs text-slate-300">
            {this.state.error?.message || "An unexpected error occurred."}
          </div>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            className="px-4 py-1.5 rounded-lg bg-indigo-600 text-white text-xs hover:bg-indigo-500 font-mono transition-colors"
          >
            Retry Workspace
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function App() {
  const [[activeTab, direction], setTabState] = useState([0, 0]);
  const [lightMode, setLightMode] = useState(false);
  const [limitationsOpen, setLimitationsOpen] = useState(false);
  const reducedMotion = useReducedMotion();

  const onSelectTab = (next) => {
    if (next !== activeTab) {
      setTabState([next, next > activeTab ? 1 : -1]);
    }
  };

  const variants = {
    enter: (d) =>
      reducedMotion
        ? { opacity: 0 }
        : { y: d > 0 ? 16 : -16, opacity: 0 },
    center: { y: 0, opacity: 1 },
    exit: (d) =>
      reducedMotion
        ? { opacity: 0 }
        : { y: d > 0 ? -16 : 16, opacity: 0 },
  };

  const tabList = [
    Tab1Corpus,
    Tab2ChampionDeberta,
    Tab3BenchmarksArchive,
    Tab4LivePredictor,
    Tab5CrossDomain,
    Tab6AiAgent,
    Tab7ErrorDiagnostics,
  ];

  const ActiveComponent = tabList[activeTab] || tabList[0];
  const currentTabMeta = WORKSPACE_TABS[activeTab] || WORKSPACE_TABS[0];

  return (
    <div className={lightMode ? "" : "dark"}>
      <div className="flex flex-row min-h-screen bg-slate-900 text-slate-100 overflow-x-hidden font-sans">
        {/* Sticky Full-Height Left Sidebar */}
        <LinuxWorkspaceBar
          activeTab={activeTab}
          onSelectTab={onSelectTab}
          lightMode={lightMode}
          onToggleLightMode={() => setLightMode(!lightMode)}
          onOpenLimitations={() => setLimitationsOpen(true)}
        />

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col justify-between min-w-0 bg-slate-950">
          <main
            role="tabpanel"
            id={`panel-${currentTabMeta.code}`}
            aria-labelledby={`tab-${currentTabMeta.code}`}
            className="max-w-[1440px] w-full mx-auto px-6 py-6 flex-1 overflow-x-hidden"
          >
            <WorkspaceErrorBoundary key={activeTab}>
              <AnimatePresence mode="wait" custom={direction}>
                <motion.div
                  key={activeTab}
                  custom={direction}
                  variants={variants}
                  initial="enter"
                  animate="center"
                  exit="exit"
                  transition={{ type: "spring", stiffness: 340, damping: 32 }}
                >
                  <ActiveComponent />
                </motion.div>
              </AnimatePresence>
            </WorkspaceErrorBoundary>
          </main>

          {/* Persistent Footer */}
          <footer className="border-t border-slate-800 bg-slate-950 py-3 px-6 mb-12">
            <div className="max-w-[1440px] mx-auto flex items-center justify-between flex-wrap gap-2 text-xs font-mono text-slate-400">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400" />
                <span>
                  Active Workspace: [{currentTabMeta.code}] {currentTabMeta.label} • Prototype (Read-Only)
                </span>
              </div>
              <button
                onClick={() => setLimitationsOpen(true)}
                className="underline text-amber-400 hover:text-amber-300 font-bold transition-colors cursor-pointer"
              >
                Open Limitations Drawer (Rule R8)
              </button>
            </div>
          </footer>
        </div>

        {/* Global Limitations Drawer */}
        <LimitationsDrawer
          open={limitationsOpen}
          onClose={() => setLimitationsOpen(false)}
        />
      </div>
    </div>
  );
}
