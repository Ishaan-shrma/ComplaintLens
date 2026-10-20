import React, { useState } from "react";
import { useData } from "../../utils/useData";
import { DataMissingPanel, MetricCard, ChartCard } from "../SharedUI";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  LabelList,
} from "recharts";
import { Trophy, Award, Sliders, Cpu, Activity, Info } from "lucide-react";

export default function Tab2ChampionDeberta() {
  const deb = useData("champion_deberta_metrics.json");
  const ofat = useData("deberta_ofat_trials.json");

  const [selectedTrialIdx, setSelectedTrialIdx] = useState(0);
  const [ofatViewMode, setOfatViewMode] = useState("bars"); // "bars" | "trajectory"

  if (deb.loading || ofat.loading) {
    return (
      <div className="p-8 font-mono text-xs text-slate-400">
        Loading champion metrics &amp; OFAT trials ...
      </div>
    );
  }

  if (deb.missing || !deb.data) {
    return <DataMissingPanel path={deb.missingPath} />;
  }

  const data = deb.data;
  const meta = data.meta || {};
  const evalSet = meta.eval_set || {};
  const cm = data.confusion_matrix || {};
  const tn = cm.tn || 0;
  const fp = cm.fp || 0;
  const fn = cm.fn || 0;
  const tp = cm.tp || 0;
  const cmTotal = tn + fp + fn + tp;

  // Recomputed metrics strictly from confusion matrix
  const recomputedAccuracy = cmTotal > 0 ? (tp + tn) / cmTotal : 0;
  const recomputedNegRecall = tn + fp > 0 ? tn / (tn + fp) : 0;
  const recomputedNegPrec = tn + fn > 0 ? tn / (tn + fn) : 0;
  const recomputedNegF1 =
    recomputedNegPrec + recomputedNegRecall > 0
      ? (2 * recomputedNegPrec * recomputedNegRecall) /
        (recomputedNegPrec + recomputedNegRecall)
      : 0;

  const recomputedPosRecall = tp + fn > 0 ? tp / (tp + fn) : 0;
  const recomputedPosPrec = tp + fp > 0 ? tp / (tp + fp) : 0;
  const recomputedPosF1 =
    recomputedPosPrec + recomputedPosRecall > 0
      ? (2 * recomputedPosPrec * recomputedPosRecall) /
        (recomputedPosPrec + recomputedPosRecall)
      : 0;

  const recomputedMacroF1 = (recomputedNegF1 + recomputedPosF1) / 2;

  // Per platform results
  const perPlatform = data.per_platform_results || [];
  const multiDomainChartData = perPlatform.map((p) => ({
    platform: p.platform,
    category: p.in_training_mix ? (p.training_role || "Replay Buffer Mix (In-Domain)") : "Held-Out Unseen Domain",
    macro_f1_pct: Number((p.macro_f1 * 100).toFixed(2)),
    negative_recall_pct: Number((p.negative_recall * 100).toFixed(2)),
    accuracy_pct: Number((p.accuracy * 100).toFixed(2)),
    in_training: p.in_training_mix,
    n: p.n,
  }));

  // OFAT trials data
  const ofatData = ofat.data || {};
  const ofatMeta = ofatData.meta || {};
  const ofatTrials = ofatData.trials || [];
  const baseline = ofatData.baseline || ofatTrials[0] || {};
  const activeTrial = ofatTrials[selectedTrialIdx] || baseline;

  const deltaF1 =
    activeTrial.macro_f1 !== undefined && baseline.macro_f1 !== undefined
      ? Number(((activeTrial.macro_f1 - baseline.macro_f1) * 100).toFixed(2))
      : 0;

  const deltaRecall =
    activeTrial.negative_recall !== undefined && baseline.negative_recall !== undefined
      ? Number(((activeTrial.negative_recall - baseline.negative_recall) * 100).toFixed(2))
      : 0;

  const ofatBarData = ofatTrials.map((t) => ({
    name: t.parameter,
    label: t.value,
    macro_f1_pct: Number((t.macro_f1 * 100).toFixed(2)),
    negative_recall_pct: Number((t.negative_recall * 100).toFixed(2)),
  }));

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-xs uppercase tracking-wider text-indigo-400 font-bold">
            [02] Champion: DeBERTa-v3-Omni
          </span>
          <span className="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-[10px] font-mono text-emerald-400 font-bold">
            #1 Champion Architecture
          </span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          Champion Model Performance &amp; OFAT Parameter Optimization
        </h2>

        {/* Section 0 Definition Banner */}
        <div className="mt-3 p-3 rounded-xl bg-slate-900 border border-indigo-500/30 flex items-start gap-2.5">
          <Award className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="text-xs font-sans">
            <span className="font-bold text-amber-300 font-mono">
              Champion Definition:{" "}
            </span>
            <span className="text-slate-200">
              {data.definition || "Best on the held-out test set among the models compared here."}
            </span>
          </div>
        </div>
      </div>

      {/* Model & Evaluation Set Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Model Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div className="text-xs font-mono text-slate-400 mb-1 flex items-center justify-between">
              <span>Model Architecture</span>
              <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="text-sm font-bold text-slate-100 font-mono truncate">
              {data.model_id}
            </div>
            <div className="text-xs text-indigo-300 font-mono mt-0.5">
              {data.model_name}
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[11px] text-slate-400 font-mono">
            {data.parameter_counts?.description || "86M backbone parameters (about 184M in total including the 128K-token embedding matrix)"}
          </div>
        </div>

        {/* Evaluation Set Card */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-sm flex flex-col justify-between">
          <div>
            <div className="text-xs font-mono text-slate-400 mb-1 flex items-center justify-between">
              <span>Evaluation Set</span>
              <Info className="w-3.5 h-3.5 text-indigo-400" />
            </div>
            <div className="text-sm font-bold text-slate-100 font-mono">
              {evalSet.name || "Canonical Held-Out Test"}
            </div>
            <div className="text-xs text-slate-400 font-mono mt-0.5">
              id: {evalSet.id} • split: {meta.split}
            </div>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800/80 text-[11px] font-mono text-slate-400">
            n = {evalSet.n?.toLocaleString()} ({evalSet.n_negative?.toLocaleString()} neg, {evalSet.n_positive?.toLocaleString()} pos)
          </div>
        </div>

        {/* Recomputed Accuracy & Macro-F1 */}
        <MetricCard
          label="Recomputed Accuracy"
          value={`${(recomputedAccuracy * 100).toFixed(2)}%`}
          sublabel="Recomputed from confusion matrix: (TP + TN) / n"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />

        <MetricCard
          label="Recomputed Macro-F1"
          value={`${(recomputedMacroF1 * 100).toFixed(2)}%`}
          sublabel="Unweighted mean of negative F1 and positive F1"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />
      </div>

      {/* Metrics Row from Confusion Matrix */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          label="Negative Recall"
          value={`${(recomputedNegRecall * 100).toFixed(2)}%`}
          sublabel="TN / (TN + FP) [Complaint Capture Rate]"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />
        <MetricCard
          label="Negative Precision"
          value={`${(recomputedNegPrec * 100).toFixed(2)}%`}
          sublabel="TN / (TN + FN) [Complaint Fidelity]"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />
        <MetricCard
          label="Missed Complaints (FP)"
          value={fp?.toLocaleString()}
          sublabel="Negative reviews predicted positive (693 vs 980 DistilBERT, -29.3%)"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />
        <MetricCard
          label="False Alarms (FN)"
          value={fn?.toLocaleString()}
          sublabel="Positive reviews predicted negative (663 vs 921 DistilBERT, -28.0%)"
          model={data.model_id}
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
        />
      </div>

      {/* Error Reduction vs DistilBERT Summary Table (Rule R4) */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-sm">
        <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-indigo-400" />
            <h3 className="text-xs md:text-sm font-bold text-slate-100 font-mono">
              Error Reduction Breakdown: DeBERTa vs DistilBERT ({evalSet.id}, n = {evalSet.n?.toLocaleString()})
            </h3>
          </div>
          <span className="text-[10px] font-mono text-emerald-400 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            {data.error_comparison_vs_distilbert?.delta_total_errors_pct}% Total Error Reduction
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                <th className="py-2 px-3">Error Classification Category</th>
                <th className="py-2 px-3">DistilBERT 281k</th>
                <th className="py-2 px-3">DeBERTa-v3-Omni</th>
                <th className="py-2 px-3">Absolute Delta</th>
                <th className="py-2 px-3">Relative Reduction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-200">
              <tr>
                <td className="py-2.5 px-3 font-semibold text-rose-300">
                  Missed Complaints (FP: Neg predicted Pos)
                </td>
                <td className="py-2.5 px-3">{data.error_comparison_vs_distilbert?.distilbert_missed_complaints_fp || 980}</td>
                <td className="py-2.5 px-3 font-bold text-emerald-400">{data.error_comparison_vs_distilbert?.deberta_missed_complaints_fp || 693}</td>
                <td className="py-2.5 px-3 text-emerald-400">-287 missed</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">{data.error_comparison_vs_distilbert?.delta_missed_complaints_pct || -29.3}%</td>
              </tr>
              <tr>
                <td className="py-2.5 px-3 font-semibold text-amber-300">
                  False Alarms (FN: Pos predicted Neg)
                </td>
                <td className="py-2.5 px-3">{data.error_comparison_vs_distilbert?.distilbert_false_alarms_fn || 921}</td>
                <td className="py-2.5 px-3 font-bold text-emerald-400">{data.error_comparison_vs_distilbert?.deberta_false_alarms_fn || 663}</td>
                <td className="py-2.5 px-3 text-emerald-400">-258 alarms</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">{data.error_comparison_vs_distilbert?.delta_false_alarms_pct || -28.0}%</td>
              </tr>
              <tr className="bg-slate-950/40 font-bold">
                <td className="py-2.5 px-3 text-slate-100">
                  Total Misclassified Reviews
                </td>
                <td className="py-2.5 px-3">{data.error_comparison_vs_distilbert?.distilbert_total_errors || 1901}</td>
                <td className="py-2.5 px-3 text-emerald-400">{data.error_comparison_vs_distilbert?.deberta_total_errors || 1356}</td>
                <td className="py-2.5 px-3 text-emerald-400">-545 total errors</td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">{data.error_comparison_vs_distilbert?.delta_total_errors_pct || -28.7}%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* Two Column Layout: Multi-Domain Chart & OFAT Studio */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Multi-Domain Performance Chart */}
        <ChartCard
          title="Multi-Domain Replay Mix Performance"
          subtitle={data.training_mix_manifest?.training_blend ? `All four platforms included in training blend (${data.training_mix_manifest.training_blend}). Macro-F1 and Negative Recall shown side by side.` : "All four platforms evaluated across training blend."}
          model={data.model_id}
          evalSetId="multi_platform"
          split={meta.split}
          n={evalSet.n}
          sourceFiles={meta.source_files}
        >
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={multiDomainChartData}
                margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="platform"
                  tick={{ fill: "#cbd5e1", fontSize: 10 }}
                  interval={0}
                />
                <YAxis
                  domain={[80, 100]}
                  tick={{ fill: "#94a3b8", fontSize: 11 }}
                />
                <Tooltip
                  formatter={(val, name) => [`${val}%`, name]}
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                />
                <Legend />
                <Bar
                  dataKey="macro_f1_pct"
                  name="Macro-F1 (%)"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                >
                  <LabelList
                    dataKey="macro_f1_pct"
                    position="top"
                    formatter={(v) => `${v}%`}
                    style={{ fill: "#cbd5e1", fontSize: 10, fontFamily: "monospace" }}
                  />
                </Bar>
                <Bar
                  dataKey="negative_recall_pct"
                  name="Negative Recall (%)"
                  fill="#10b981"
                  radius={[4, 4, 0, 0]}
                >
                  <LabelList
                    dataKey="negative_recall_pct"
                    position="top"
                    formatter={(v) => `${v}%`}
                    style={{ fill: "#cbd5e1", fontSize: 10, fontFamily: "monospace" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 pt-2 border-t border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
            {multiDomainChartData.map((p) => (
              <div key={p.platform} className="flex items-center justify-between">
                <span>{p.platform}:</span>
                <span className={p.in_training ? "text-indigo-400" : "text-amber-400 font-semibold"}>
                  {p.category} (n = {p.n?.toLocaleString()})
                </span>
              </div>
            ))}
          </div>
        </ChartCard>

        {/* OFAT Studio */}
        <ChartCard
          title="OFAT Hyperparameter Optimization Studio"
          subtitle="Tuning runs evaluated strictly on validation split (Rule R5). Delta computed against baseline trial."
          model={data.model_id}
          evalSetId={ofatMeta.eval_set?.id || "amazon_val"}
          split={ofatMeta.split || "validation"}
          n={ofatMeta.eval_set?.n}
          sourceFiles={ofatMeta.source_files}
        >
          {/* Validation Split Badge (R5) */}
          <div className="flex items-center justify-between mb-3">
            <span className="px-2 py-0.5 rounded bg-amber-500/10 border border-amber-500/20 text-[10px] font-mono text-amber-300 font-bold">
              split: {ofatMeta.split || "validation"} (Rule R5 Enforced)
            </span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => setOfatViewMode("bars")}
                className={`px-2.5 py-1 rounded text-[11px] font-mono border ${
                  ofatViewMode === "bars"
                    ? "bg-indigo-600 border-indigo-500 text-white"
                    : "border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                Metric Bars
              </button>
              <button
                onClick={() => setOfatViewMode("trajectory")}
                className={`px-2.5 py-1 rounded text-[11px] font-mono border ${
                  ofatViewMode === "trajectory"
                    ? "bg-indigo-600 border-indigo-500 text-white"
                    : "border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                2-Epoch Trajectories
              </button>
            </div>
          </div>

          {/* Trial Pills */}
          <div className="flex flex-wrap gap-1.5 mb-3">
            {ofatTrials.map((tr, idx) => (
              <button
                key={tr.trial_id || idx}
                onClick={() => setSelectedTrialIdx(idx)}
                className={`px-2.5 py-1 rounded-lg text-[11px] font-mono border transition-colors ${
                  selectedTrialIdx === idx
                    ? "bg-indigo-500/20 border-indigo-400 text-indigo-300 font-bold"
                    : "border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700"
                }`}
              >
                {tr.parameter}: {tr.value}
              </button>
            ))}
          </div>

          {/* Active Trial Delta Cards */}
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 font-mono">
              <div className="text-[10px] text-slate-400 uppercase">
                Delta vs Baseline Macro-F1
              </div>
              <div
                className={`text-lg font-bold ${
                  deltaF1 >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {deltaF1 >= 0 ? `+${deltaF1}%` : `${deltaF1}%`}
              </div>
              <div className="text-[10px] text-slate-500 truncate">
                Current: {(activeTrial.macro_f1 * 100).toFixed(2)}% | Base: {(baseline.macro_f1 * 100).toFixed(2)}%
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 font-mono">
              <div className="text-[10px] text-slate-400 uppercase">
                Delta vs Baseline Neg Recall
              </div>
              <div
                className={`text-lg font-bold ${
                  deltaRecall >= 0 ? "text-emerald-400" : "text-rose-400"
                }`}
              >
                {deltaRecall >= 0 ? `+${deltaRecall}%` : `${deltaRecall}%`}
              </div>
              <div className="text-[10px] text-slate-500 truncate">
                Current: {(activeTrial.negative_recall * 100).toFixed(2)}% | Base: {(baseline.negative_recall * 100).toFixed(2)}%
              </div>
            </div>
          </div>

          {/* Chart Display: Bars or Trajectory */}
          {ofatViewMode === "bars" ? (
            <div className="h-52">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={ofatBarData}
                  margin={{ top: 15, right: 15, left: 0, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="name" tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                  <YAxis domain={[90, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip
                    formatter={(val, name) => [`${val}%`, name]}
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                  />
                  <Bar
                    dataKey="macro_f1_pct"
                    name="Val Macro-F1 (%)"
                    fill="#6366f1"
                    radius={[4, 4, 0, 0]}
                  >
                    <LabelList
                      dataKey="macro_f1_pct"
                      position="top"
                      formatter={(v) => `${v}%`}
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
                    />
                  </Bar>
                  <Bar
                    dataKey="negative_recall_pct"
                    name="Val Neg Recall (%)"
                    fill="#10b981"
                    radius={[4, 4, 0, 0]}
                  >
                    <LabelList
                      dataKey="negative_recall_pct"
                      position="top"
                      formatter={(v) => `${v}%`}
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="h-52">
              {activeTrial.epoch_log && activeTrial.epoch_log.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={activeTrial.epoch_log.map((ep) => ({
                      epoch: `Epoch ${ep.epoch}`,
                      val_f1_pct: Number((ep.val_f1 * 100).toFixed(2)),
                      val_loss: ep.val_loss,
                    }))}
                    margin={{ top: 15, right: 20, left: 0, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                    <XAxis dataKey="epoch" tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                    <YAxis domain={[95, 98]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                    <Tooltip
                      formatter={(val, name) => [
                        name === "Val Macro-F1 (%)" ? `${val}%` : val,
                        name,
                      ]}
                      contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                    />
                    <Line
                      type="monotone"
                      dataKey="val_f1_pct"
                      name="Val Macro-F1 (%)"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={{ fill: "#6366f1", r: 4 }}
                    >
                      <LabelList
                        dataKey="val_f1_pct"
                        position="top"
                        formatter={(v) => `${v}%`}
                        style={{ fill: "#cbd5e1", fontSize: 10, fontFamily: "monospace" }}
                      />
                    </Line>
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="p-8 text-center text-xs font-mono text-slate-500">
                  No multi-epoch trajectory logged for trial {activeTrial.trial_id}
                </div>
              )}
            </div>
          )}
        </ChartCard>
      </div>
    </div>
  );
}
