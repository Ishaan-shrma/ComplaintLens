import React, { useState } from "react";
import { useData } from "../../utils/useData";
import { DataMissingPanel, MetricCard, ChartCard } from "../SharedUI";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList,
} from "recharts";
import { Bug, FileCheck, Layers, Quote, Info, AlertTriangle, ShieldCheck } from "lucide-react";

export default function Tab7ErrorDiagnostics() {
  const errData = useData("error_taxonomy.json");
  const [selectedCatId, setSelectedCatId] = useState(null);

  if (errData.loading) {
    return (
      <div className="p-8 font-mono text-xs text-slate-400">
        Loading error taxonomy and diagnostics ...
      </div>
    );
  }

  if (errData.missing || !errData.data) {
    return <DataMissingPanel path={errData.missingPath || "error_taxonomy.json"} />;
  }

  const data = errData.data;
  const meta = data.meta || {};
  const evalSet = meta.eval_set || {};
  const comparison = data.comparison || {};
  const overlap = data.overlap || {};
  const categories = data.categories || [];

  if (!categories || categories.length === 0 || !data.n_errors_total) {
    return <DataMissingPanel path="error_taxonomy.json (categories/n_errors_total)" />;
  }

  const activeCategory =
    categories.find((c) => c.id === selectedCatId) || categories[0] || {};

  // Prepare horizontal bar data (showing both DeBERTa primary and DistilBERT comparison counts)
  const chartData = categories.map((c) => ({
    id: c.id,
    label: c.label,
    deberta: c.count_tagged ?? c.count,
    distilbert: c.distilbert_count ?? 0,
    deberta_share: c.share != null ? Number((c.share * 100).toFixed(1)) : 0,
    distilbert_share: c.distilbert_share != null ? Number((c.distilbert_share * 100).toFixed(1)) : 0,
    total_reviews: c.total_reviews,
    error_rate_pct: c.error_rate_pct,
  }));

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-xs uppercase tracking-wider text-indigo-400 font-bold">
            [07] Error Taxonomy and Diagnostics
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-400">
            error_taxonomy.json
          </span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          Model Failure Mode Taxonomy &amp; Error Diagnostics
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Rule-based categorization of held-out test errors to diagnose syntactic blind spots, rating inversions, and sequence budget truncation.
        </p>

        {/* Error Definitions Banner */}
        <div className="mt-3 p-3 rounded-xl border border-slate-800 bg-slate-900/60 font-mono text-xs text-slate-300 flex items-center gap-2">
          <Info className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>
            <strong className="text-slate-100">Error Definitions:</strong> Missed complaint = actual negative predicted positive (FP); False alarm = actual positive predicted negative (FN).
          </span>
        </div>
      </div>

      {/* Audit Meta & Scope Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Primary Model"
          value={data.display_name || data.model_id}
          sublabel={
            data.champion_definition
              ? `${data.champion_definition} • Comparison: ${comparison.display_name || "DistilBERT 281k"}`
              : "Best on held-out test set among models compared here • Comparison: DistilBERT 281k"
          }
          model={data.model_id}
          config={data.inference_config || "raw text, argmax"}
          evalSetId={data.eval_set_id || evalSet.id}
          split={data.split || meta.split}
          n={data.n || evalSet.n}
        />
        <MetricCard
          label="Total Test Errors"
          value={data.n_errors_total?.toLocaleString()}
          sublabel={
            comparison.n_errors_total
              ? `Comparison model: ${comparison.n_errors_total.toLocaleString()} errors`
              : "Held-out test split"
          }
          model={data.model_id}
          config={data.inference_config || "raw text, argmax"}
          evalSetId={data.eval_set_id || evalSet.id}
          split={data.split || meta.split}
          n={data.n_errors_total}
        />
        <MetricCard
          label="Errors Tagged"
          value={data.n_tagged?.toLocaleString()}
          sublabel={data.labeling_type || "Rule-based categories"}
          model={data.model_id}
          config={data.inference_config || "raw text, argmax"}
          evalSetId={data.eval_set_id || evalSet.id}
          split={data.split || meta.split}
          n={data.n_tagged}
        />
        <div
          className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-sm flex flex-col justify-between"
          title={`Fixed: ${overlap.fixed_definition || "DistilBERT wrong, DeBERTa correct"} • New: ${overlap.new_definition || "DeBERTa wrong, DistilBERT correct"} • Shared: ${overlap.shared_definition || "Both models wrong"}`}
        >
          <div>
            <div className="text-xs font-mono text-slate-400 mb-1 flex items-center justify-between">
              <span>Categorization Method</span>
              <FileCheck className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="text-xs font-bold text-slate-200 font-mono capitalize">
              {data.labeling_type || "Rule-Based Categories"}
            </div>
            <div className="text-[10px] text-indigo-300 font-mono mt-1">
              {overlap.shared_errors != null && overlap.new_errors != null
                ? `${overlap.shared_errors} shared • ${overlap.new_errors} new • ${overlap.fixed_errors} fixed`
                : "Deterministic priority rule tagging"}
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[10px] font-mono text-slate-500 space-y-1">
            <div title={overlap.fixed_definition}>
              <strong className="text-slate-400">Fixed ({overlap.fixed_errors}):</strong> {overlap.fixed_definition}
            </div>
            <div title={overlap.new_definition}>
              <strong className="text-slate-400">New ({overlap.new_errors}):</strong> {overlap.new_definition}
            </div>
            <div title={overlap.shared_definition}>
              <strong className="text-slate-400">Shared ({overlap.shared_errors}):</strong> {overlap.shared_definition}
            </div>
          </div>
        </div>
      </div>

      {/* Adaptive Reuse Disclosure Banner */}
      {data.disclosure && (
        <div className="p-3.5 rounded-xl border border-amber-500/30 bg-amber-950/15 font-mono text-xs text-amber-300/90 flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <span className="font-bold text-amber-200">Methodological Disclosure:</span>
            <p className="font-sans text-[11px] text-slate-300 leading-relaxed">
              {data.disclosure}
            </p>
          </div>
        </div>
      )}

      {/* Two Column Layout: Horizontal Bar Chart vs Category Deep-Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Horizontal Bar Chart comparing DeBERTa primary and DistilBERT comparison */}
        <div className="lg:col-span-8">
          <ChartCard
            title={data.chart_title || "Misclassification Distribution by Rule-Based Category"}
            subtitle={
              data.chart_subtitle ||
              "Dual-model comparison of discrete error counts. Indigo: DeBERTa-v3-Omni (n = 1,382 errors, primary); Grey: DistilBERT 281k (n = 1,901 errors, comparison)."
            }
            model={data.model_id}
            config={data.inference_config || "raw text, argmax"}
            evalSetId={data.eval_set_id || evalSet.id}
            split={data.split || meta.split}
            n={data.n_tagged}
            sourceFiles={meta.source_files}
          >
            <div style={{ height: 460 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  layout="vertical"
                  margin={{ top: 15, right: 45, left: 10, bottom: 15 }}
                  onClick={(state) => {
                    if (state && state.activePayload && state.activePayload.length > 0) {
                      setSelectedCatId(state.activePayload[0].payload.id);
                    }
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    type="number"
                    tick={{ fill: "#94a3b8", fontSize: 10 }}
                  />
                  <YAxis
                    dataKey="label"
                    type="category"
                    tick={{ fill: "#cbd5e1", fontSize: 10 }}
                    width={250}
                  />
                  <Tooltip
                    formatter={(val, name) => [
                      typeof val === "number" ? val.toLocaleString() : val,
                      name,
                    ]}
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                  />
                  <Legend verticalAlign="top" height={36} />
                  <Bar
                    dataKey="deberta"
                    name={data.chart_primary_legend || "DeBERTa-v3-Omni (Primary, n = 1,382)"}
                    fill="#6366f1"
                    radius={[0, 4, 4, 0]}
                  >
                    <LabelList
                      dataKey="deberta"
                      position="right"
                      formatter={(v) => (typeof v === "number" ? v.toLocaleString() : v)}
                      style={{ fill: "#cbd5e1", fontSize: 10, fontFamily: "monospace" }}
                    />
                  </Bar>
                  <Bar
                    dataKey="distilbert"
                    name={data.chart_comparison_legend || "DistilBERT 281k (Comparison, n = 1,901)"}
                    fill="#64748b"
                    radius={[0, 4, 4, 0]}
                  >
                    <LabelList
                      dataKey="distilbert"
                      position="right"
                      formatter={(v) => (typeof v === "number" ? v.toLocaleString() : v)}
                      style={{ fill: "#94a3b8", fontSize: 10, fontFamily: "monospace" }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-[10px] font-mono text-slate-500 flex flex-col md:flex-row md:items-center justify-between gap-1">
              <span>Note: Click any bar to inspect category diagnostics, comparative rates, and verified example review IDs.</span>
              {data.chart_footer_note && (
                <span className="text-slate-400">{data.chart_footer_note}</span>
              )}
            </div>
          </ChartCard>
        </div>

        {/* Category Examples and Forensic Inspection */}
        <div className="lg:col-span-4 space-y-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm space-y-4">
            <div className="border-b border-slate-800 pb-3">
              <span className="text-[10px] font-mono uppercase text-indigo-400 font-bold">
                Selected Failure Mode
              </span>
              <h3 className="text-sm font-bold text-slate-100 font-mono mt-0.5">
                {activeCategory.label}
              </h3>
              {activeCategory.description && (
                <p className="text-[11px] text-slate-400 font-sans mt-1 leading-relaxed">
                  {activeCategory.description}
                </p>
              )}
            </div>

            {/* Error Metrics: Both models' counts, total reviews, and error rates */}
            <div className="grid grid-cols-2 gap-3 font-mono text-xs">
              <div className="p-3 rounded-xl border border-indigo-500/30 bg-indigo-950/20">
                <div className="text-[10px] text-indigo-300">DeBERTa Errors</div>
                <div className="text-base font-bold text-indigo-200 mt-0.5">
                  {activeCategory.count?.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-400 mt-0.5">
                  {activeCategory.share != null ? `${(activeCategory.share * 100).toFixed(1)}% of DeBERTa` : ""}
                </div>
              </div>

              <div className="p-3 rounded-xl border border-slate-800 bg-slate-950/70">
                <div className="text-[10px] text-slate-400">DistilBERT Errors</div>
                <div className="text-base font-bold text-slate-300 mt-0.5">
                  {activeCategory.distilbert_count?.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  {activeCategory.distilbert_share != null ? `${(activeCategory.distilbert_share * 100).toFixed(1)}% of DistilBERT` : ""}
                </div>
              </div>

              <div className="p-3 rounded-xl border border-slate-800 bg-slate-950/70">
                <div className="text-[10px] text-slate-400">Total in Category</div>
                <div className="text-base font-bold text-slate-100 mt-0.5">
                  {activeCategory.total_reviews?.toLocaleString()}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  All test reviews
                </div>
              </div>

              <div className="p-3 rounded-xl border border-slate-800 bg-slate-950/70">
                <div className="text-[10px] text-slate-400">DeBERTa Error Rate</div>
                <div className="text-base font-bold text-emerald-400 mt-0.5">
                  {activeCategory.error_rate_pct != null ? `${activeCategory.error_rate_pct}%` : "N/A"}
                </div>
                <div className="text-[10px] text-slate-500 mt-0.5">
                  {activeCategory.distilbert_error_rate_pct != null ? `DistilBERT: ${activeCategory.distilbert_error_rate_pct}%` : ""}
                </div>
              </div>
            </div>

            {/* Note under Mixed / Contrastive Clauses row */}
            {activeCategory.id === "mixed_clauses" && (
              <div className="p-2.5 rounded-lg border border-slate-800 bg-slate-950/80 font-mono text-[10px] text-slate-400">
                <span className="text-indigo-400 font-bold">Rule Note:</span> The mixed/contrastive rule is broad and also applies to many correctly classified reviews.
              </div>
            )}

            {/* Comparative Error Reduction Note (Rule R6: Associated with, no unablated causal claims) */}
            {activeCategory.comparative_note && (
              <div className="p-3 rounded-xl border border-indigo-500/20 bg-indigo-950/20 font-mono text-[11px] text-indigo-300 space-y-1">
                <div className="font-bold text-indigo-200 flex items-center gap-1.5">
                  <Info className="w-3 h-3 text-indigo-400" />
                  <span>Comparative Diagnostics</span>
                </div>
                <p className="text-slate-300 font-sans text-[11px] leading-relaxed">
                  {activeCategory.comparative_note}
                </p>
              </div>
            )}

            {/* Example Review IDs from DeBERTa Errors */}
            <div className="space-y-2 pt-2 border-t border-slate-800 font-mono">
              <div className="flex items-center gap-2 text-xs font-bold text-slate-200">
                <Quote className="w-3.5 h-3.5 text-indigo-400" />
                <span>Verified DeBERTa Error Review IDs</span>
              </div>
              <div className="space-y-1.5">
                {(activeCategory.example_ids || []).map((idStr, idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-xl border border-slate-800 bg-slate-950/80 flex items-center justify-between text-xs"
                  >
                    <span className="text-slate-400">Review ID:</span>
                    <span className="text-indigo-400 font-bold bg-indigo-500/10 px-2 py-0.5 rounded border border-indigo-500/20">
                      #{idStr}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
