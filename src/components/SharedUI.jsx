import React from "react";
import { AlertTriangle, FileQuestion } from "lucide-react";

export function DataMissingPanel({ path, missingKey, error }) {
  const displayPath = path || "unknown/path";
  return (
    <div className="rounded-2xl border border-rose-500/30 bg-rose-950/20 p-8 text-center backdrop-blur-md">
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-rose-500/20 text-rose-400">
        <FileQuestion className="h-6 w-6" />
      </div>
      <h3 className="text-base font-semibold text-rose-300">
        {missingKey ? `Missing Key: ${missingKey}` : "Data File Missing"}
      </h3>
      <p className="mt-2 font-mono text-xs text-rose-400 bg-black/50 px-3 py-1.5 rounded-lg inline-block border border-rose-500/20">
        data file not found: {displayPath}
      </p>
      {error && (
        <p className="mt-2 text-xs text-slate-400">Reason: {String(error)}</p>
      )}
      <div className="mt-4 flex items-center justify-center gap-2 text-xs text-slate-500 font-mono">
        <AlertTriangle className="h-3.5 w-3.5 text-amber-400 shrink-0" />
        <span>Rule R7: Missing data is shown as missing. Fabricated values are strictly prohibited.</span>
      </div>
    </div>
  );
}

export function MetricCard({
  label,
  value,
  sublabel,
  model,
  evalSetId,
  split,
  n,
  config,
  pathForMissing,
}) {
  const hasContext = Boolean(model || evalSetId || split || n || config);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-sm flex flex-col justify-between">
      <div>
        <div className="text-xs font-mono text-slate-400 mb-1">{label}</div>
        <div className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          {value !== undefined && value !== null ? (
            value
          ) : (
            <span className="text-rose-400 text-xs font-mono">
              data file not found: {pathForMissing || "unspecified"}
            </span>
          )}
        </div>
      </div>

      <div className="space-y-1 mt-3 pt-2 border-t border-slate-800/80">
        {sublabel && (
          <div className="text-[11px] font-sans text-slate-400 line-clamp-2">
            {sublabel}
          </div>
        )}
        {hasContext && (
          <div className="text-[10px] font-mono text-slate-500 truncate">
            {[
              model,
              config,
              evalSetId,
              split,
              n ? `n = ${typeof n === "number" ? n.toLocaleString() : n}` : null,
            ]
              .filter(Boolean)
              .join(" • ")}
          </div>
        )}
      </div>
    </div>
  );
}

export function ChartCard({
  title,
  subtitle,
  model,
  evalSetId,
  split,
  n,
  config,
  sourceFiles,
  children,
}) {
  const contextParts = [
    model,
    config,
    evalSetId,
    split,
    n ? `n = ${typeof n === "number" ? n.toLocaleString() : n}` : null,
  ].filter(Boolean);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 md:p-6 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3 border-b border-slate-800 pb-3 mb-4">
          <div>
            <h3 className="text-sm md:text-base font-bold text-slate-100 font-mono">
              {title}
            </h3>
            {subtitle && (
              <p className="text-xs text-slate-400 mt-1 font-sans leading-relaxed">
                {subtitle}
              </p>
            )}
          </div>

          {contextParts.length > 0 && (
            <div className="flex items-center gap-1.5 flex-wrap font-mono text-[10px] shrink-0">
              {split && (
                <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
                  {split}
                </span>
              )}
              {n && (
                <span className="px-2 py-0.5 rounded bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 font-bold">
                  n = {typeof n === "number" ? n.toLocaleString() : n}
                </span>
              )}
            </div>
          )}
        </div>

        <div className="w-full">{children}</div>
      </div>

      <div className="mt-4 pt-3 border-t border-slate-800/80 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px] text-slate-500">
        <div>
          {contextParts.length > 0 && (
            <span>Context: {contextParts.join(", ")}</span>
          )}
        </div>
        {sourceFiles && sourceFiles.length > 0 && (
          <span>Source: {sourceFiles.filter(Boolean).join(", ")}</span>
        )}
      </div>
    </div>
  );
}
