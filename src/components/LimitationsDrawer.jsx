import React, { useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldAlert, ChevronUp, ChevronDown, Info, X } from "lucide-react";

export default function LimitationsDrawer({ open, onClose }) {
  const [internalOpen, setInternalOpen] = useState(false);
  const isDrawerOpen = open || internalOpen;

  const handleClose = () => {
    if (onClose) onClose();
    setInternalOpen(false);
  };

  const toggleDrawer = () => {
    if (isDrawerOpen) {
      handleClose();
    } else {
      setInternalOpen(true);
    }
  };

  const limitations = [
    {
      title: "Domain & Temporal Coverage",
      statement: "Reviews cover four platforms; the food data ends in 2012. Results may not transfer to other domains or periods.",
    },
    {
      title: "Test-Set Adaptive Feedback Disclosure",
      statement: "DeBERTa-v3-Omni's modeling choices were informed by error patterns audited on DistilBERT's held-out test predictions. This represents adaptive feedback from the test split, warranting validation on fresh temporal or external cohorts to guard against test-set overfitting.",
    },
    {
      title: "Multi-Domain Replay Buffer Composition",
      statement: "DeBERTa-v3-Omni was trained on an 85/15 blend (281k Amazon + 42k Omni-corpus replay from Flipkart, Yelp, and IMDb). Cross-platform performance reflects domain-adapted fine-tuning rather than zero-shot transfer.",
    },
    {
      title: "Label Noise & Neutral Exclusion",
      statement: "Labels come from star ratings or each platform's own labels; they contain noise. Neutral (3-star) reviews are excluded from the binary models.",
    },
    {
      title: "Amazon Pool Class Imbalance",
      statement: "Class imbalance is about 84% positive and 16% negative in the Amazon pool.",
    },
    {
      title: "Sequence Truncation",
      statement: "Inputs are truncated to a fixed token limit.",
    },
    {
      title: "Language Scope",
      statement: "English-language models; non-English text is not supported.",
    },
    {
      title: "Retrieval Index Sampling",
      statement: "The embedding index covers a sample, not the full corpus.",
    },
    {
      title: "Prototype Scope",
      statement: "Prototype: not validated for production use.",
    },
  ];

  return (
    <aside aria-label="System Limitations">
      {/* Sticky Bottom Bar */}
      <footer className="fixed bottom-0 left-0 md:left-64 lg:left-72 right-0 z-40 border-t border-slate-800 bg-slate-950/95 backdrop-blur-xl text-slate-400 shadow-2xl">
        <div className="w-full px-6 lg:px-8">
          <div className="flex h-12 items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-mono text-[11px] uppercase tracking-wider text-slate-300">
                Review Intelligence Prototype (Read-Only)
              </span>
              <span className="text-slate-700">|</span>
              <span className="text-[11px] text-slate-500 hidden sm:inline">
                Four Review Corpora • Multi-Platform
              </span>
            </div>

            <button
              onClick={toggleDrawer}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-800/80 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-slate-200 hover:border-slate-500 hover:text-white transition-all shadow-sm cursor-pointer"
            >
              <ShieldAlert className="h-3.5 w-3.5 text-amber-400" />
              <span>Section 9: Limitations Drawer</span>
              {isDrawerOpen ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronUp className="h-3.5 w-3.5" />
              )}
            </button>
          </div>
        </div>

        {/* Collapsible Panel */}
        <AnimatePresence>
          {isDrawerOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.25, ease: "easeInOut" }}
              className="overflow-hidden border-t border-slate-800 bg-slate-900/98 backdrop-blur-2xl"
            >
              <div className="w-full px-6 lg:px-8 py-6 max-h-[70vh] overflow-y-auto">
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Info className="h-4 w-4 text-indigo-400" />
                    <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-200 font-mono">
                      Engineering Limitations &amp; Scope Disclosure (Rule R8)
                    </h4>
                  </div>
                  <button
                    onClick={handleClose}
                    className="p-1 rounded-md text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
                    aria-label="Close limitations drawer"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                  {limitations.map((item, idx) => (
                    <div
                      key={idx}
                      className="rounded-xl border border-slate-800 bg-slate-950/70 p-4 transition-colors hover:border-slate-700"
                    >
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="flex h-5 w-5 items-center justify-center rounded bg-indigo-500/10 font-mono text-[10px] text-indigo-400 font-bold border border-indigo-500/20">
                          {idx + 1}
                        </span>
                        <h5 className="text-xs font-semibold text-slate-200">
                          {item.title}
                        </h5>
                      </div>
                      <p className="text-[11px] leading-relaxed text-slate-400">
                        {item.statement}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </footer>
    </aside>
  );
}
