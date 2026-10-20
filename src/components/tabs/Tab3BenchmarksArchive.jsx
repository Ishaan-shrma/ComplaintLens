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
  ReferenceLine,
} from "recharts";
import {
  BarChart3,
  Scale,
  Activity,
  Sliders,
  CheckCircle,
  AlertTriangle,
  Layers,
} from "lucide-react";

export default function Tab3BenchmarksArchive() {
  const comp = useData("model_comparison.json");
  const distilMulti = useData("distilbert_multiparams.json");
  const trainCurve = useData("training_curve.json");
  const calib = useData("calibration_curve.json");

  const [leaderboardView, setLeaderboardView] = useState("absolute"); // "absolute" | "delta"
  const [selectedCalibModel, setSelectedCalibModel] = useState("distilbert"); // "distilbert" | "deberta"
  const [calibT, setCalibT] = useState(null);

  if (comp.loading || distilMulti.loading || trainCurve.loading || calib.loading) {
    return (
      <div className="p-8 font-mono text-xs text-slate-400">
        Loading benchmarks archive, training curves &amp; calibration grids ...
      </div>
    );
  }

  if (comp.missing || !comp.data) {
    return <DataMissingPanel path={comp.missingPath} />;
  }

  const compData = comp.data;
  const compMeta = compData.meta || {};
  const canonicalEvalId = compMeta.eval_set?.id || "amazon_test";
  const rows = compData.rows || [];

  // Find champion row
  const championRow = rows.find((r) => r.eval_set_id === canonicalEvalId) || rows[0];

  // DistilBERT studio data
  const distilData = distilMulti.data || {};
  const scaleRuns = distilData.scale_runs || [];
  const weightRuns = distilData.weight_runs || [];
  const testDiag = distilData.test_diagnostic || {};

  // Training curve data
  const trainHistory = (trainCurve.data?.history || []).map((h) => ({
    step: h.step,
    epoch: h.epoch,
    train_loss: h.train_loss,
    eval_loss: h.eval_loss,
  }));

  // Calibration curve data
  const calibData = calib.data?.models || {};
  const activeCalibModel = calibData[selectedCalibModel] || calibData["distilbert"] || {};
  const defaultTKey = activeCalibModel.fitted_T ? activeCalibModel.fitted_T.toFixed(1) : "1.7";
  const currentTKey = calibT !== null ? calibT : defaultTKey;
  const currentGrid = activeCalibModel.grid?.[currentTKey] || activeCalibModel.grid?.[defaultTKey] || {};
  const reliabilityBins = currentGrid.bins || [];

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-xs uppercase tracking-wider text-indigo-400 font-bold">
            [03] Benchmarks and Model Archive
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-400">
            model_comparison.json
          </span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          Model Leaderboard, Hyperparameter Sweeps &amp; Reliability Calibration
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Rigorous cross-model evaluations, two-facet DistilBERT scale/weight sweeps, HuggingFace training trajectories, and post-hoc temperature scaling.
        </p>
      </div>

      {/* 1. Model Leaderboard with R4 Enforcement */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-slate-800 pb-3">
          <div>
            <div className="flex items-center gap-2">
              <BarChart3 className="w-4 h-4 text-indigo-400" />
              <h3 className="text-sm md:text-base font-bold text-slate-100 font-mono">
                Model Comparison Leaderboard
              </h3>
            </div>
            <p className="text-xs text-slate-400 mt-0.5">
              Rule R4 enforced: Deltas and rankings compare strictly like-with-like (same eval_set.id). Different sets are greyed out.
            </p>
          </div>

          {/* Toggle Absolute vs Delta */}
          <div className="flex items-center gap-1 font-mono text-xs shrink-0">
            <button
              onClick={() => setLeaderboardView("absolute")}
              className={`px-3 py-1.5 rounded-lg border transition-colors ${
                leaderboardView === "absolute"
                  ? "bg-indigo-600 border-indigo-500 text-white font-bold"
                  : "border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              Absolute (%)
            </button>
            <button
              onClick={() => setLeaderboardView("delta")}
              className={`px-3 py-1.5 rounded-lg border transition-colors ${
                leaderboardView === "delta"
                  ? "bg-indigo-600 border-indigo-500 text-white font-bold"
                  : "border-slate-800 text-slate-400 hover:text-slate-200"
              }`}
            >
              Change vs Champion
            </button>
          </div>
        </div>

        {/* Leaderboard Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">Training n</th>
                <th className="py-2.5 px-3">Eval Set</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Macro-F1</th>
                <th className="py-2.5 px-3">Neg Recall</th>
                <th className="py-2.5 px-3">Neg Precision</th>
                <th className="py-2.5 px-3">FP / FN</th>
                <th className="py-2.5 px-3">Evaluation Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {rows.map((r) => {
                const isComparable = r.eval_set_id === canonicalEvalId;
                const isChamp = r.model_id === compData.champion_model_id;

                const deltaAcc = isComparable
                  ? Number(((r.accuracy - championRow.accuracy) * 100).toFixed(2))
                  : null;
                const deltaF1 = isComparable
                  ? Number(((r.macro_f1 - championRow.macro_f1) * 100).toFixed(2))
                  : null;
                const deltaRec = isComparable
                  ? Number(((r.negative_recall - championRow.negative_recall) * 100).toFixed(2))
                  : null;

                return (
                  <tr
                    key={r.model_id}
                    className={`transition-colors ${
                      !isComparable
                        ? "bg-slate-950/40 text-slate-500 opacity-60"
                        : isChamp
                        ? "bg-indigo-950/30 font-semibold text-slate-100"
                        : "hover:bg-slate-800/40 text-slate-300"
                    }`}
                  >
                    <td className="py-3 px-3 font-semibold">
                      <div className="flex items-center gap-1.5">
                        {isChamp && (
                          <span className="w-2 h-2 rounded-full bg-emerald-400" />
                        )}
                        <span>{r.display_name || r.model_id}</span>
                      </div>
                    </td>
                    <td className="py-3 px-3">
                      {r.train_n ? r.train_n.toLocaleString() : "N/A (Pretrained)"}
                    </td>
                    <td className="py-3 px-3 text-slate-400">{r.eval_set_id}</td>

                    {/* Accuracy Column */}
                    <td className="py-3 px-3">
                      {leaderboardView === "absolute" ? (
                        `${(r.accuracy * 100).toFixed(2)}%`
                      ) : !isComparable ? (
                        <span className="text-slate-500 italic">different test set, not comparable</span>
                      ) : isChamp ? (
                        <span className="text-indigo-400">Champion Base</span>
                      ) : (
                        <span className={deltaAcc >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          {deltaAcc >= 0 ? `+${deltaAcc}%` : `${deltaAcc}%`}
                        </span>
                      )}
                    </td>

                    {/* Macro-F1 Column */}
                    <td className="py-3 px-3">
                      {leaderboardView === "absolute" ? (
                        `${(r.macro_f1 * 100).toFixed(2)}%`
                      ) : !isComparable ? (
                        <span className="text-slate-500 italic">different test set, not comparable</span>
                      ) : isChamp ? (
                        <span className="text-indigo-400">Champion Base</span>
                      ) : (
                        <span className={deltaF1 >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          {deltaF1 >= 0 ? `+${deltaF1}%` : `${deltaF1}%`}
                        </span>
                      )}
                    </td>

                    {/* Negative Recall Column */}
                    <td className="py-3 px-3">
                      {leaderboardView === "absolute" ? (
                        `${(r.negative_recall * 100).toFixed(2)}%`
                      ) : !isComparable ? (
                        <span className="text-slate-500 italic">different test set, not comparable</span>
                      ) : isChamp ? (
                        <span className="text-indigo-400">Champion Base</span>
                      ) : (
                        <span className={deltaRec >= 0 ? "text-emerald-400" : "text-rose-400"}>
                          {deltaRec >= 0 ? `+${deltaRec}%` : `${deltaRec}%`}
                        </span>
                      )}
                    </td>

                    {/* Negative Precision */}
                    <td className="py-3 px-3">
                      {r.negative_precision ? `${(r.negative_precision * 100).toFixed(2)}%` : "N/A"}
                    </td>

                    {/* FP / FN */}
                    <td className="py-3 px-3">
                      {r.fp !== undefined && r.fn !== undefined
                        ? `${r.fp.toLocaleString()} FP / ${r.fn.toLocaleString()} FN`
                        : "N/A"}
                    </td>

                    {/* Status (Rule R5: evaluated once) */}
                    <td className="py-3 px-3">
                      <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-300">
                        {r.status || "evaluated once"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. DistilBERT Studio as Two Linked Facets */}
      <div className="space-y-4">
        <div className="border-b border-slate-800 pb-2">
          <h3 className="text-base font-bold text-slate-100 font-mono flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-400" />
            DistilBERT Optimization Studio (Two Linked Facets)
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Facet (a) Training scale 50k vs 281k unweighted; Facet (b) Class-weight sweep 1.0x to 5.4x (run on 50k subset only).
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Facet A: Training Set Size */}
          <ChartCard
            title="Facet (a): Training Set Scale (50k vs 281k, Unweighted)"
            subtitle="Comparing data scaling efficiency on identical unweighted architecture"
            model="distilbert-base-uncased"
            evalSetId={distilData.meta?.eval_set?.id || "amazon_val"}
            split={distilData.meta?.split || "validation"}
            n={281271}
            sourceFiles={distilData.meta?.source_files}
          >
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={scaleRuns.map((s) => ({
                    label: `Train n=${s.train_n?.toLocaleString()}`,
                    macro_f1_pct: Number((s.macro_f1 * 100).toFixed(2)),
                    negative_recall_pct: Number((s.negative_recall * 100).toFixed(2)),
                    accuracy_pct: Number((s.accuracy * 100).toFixed(2)),
                    train_n: s.train_n,
                  }))}
                  margin={{ top: 20, right: 20, left: 0, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="label" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                  <YAxis domain={[85, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
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
                    name="Neg Recall (%)"
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
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              Training scaling from 50k to 281k reviews associated with an improvement in unweighted Macro-F1.
            </div>
          </ChartCard>

          {/* Facet B: Class-Weight Sweep */}
          <ChartCard
            title="Facet (b): Class-Weight Sweep (50k Subset Only)"
            subtitle="Loss weight sweep from 1.0x to 5.4x inverse-frequency on 50k training records"
            model="distilbert-base-uncased"
            evalSetId={distilData.meta?.eval_set?.id || "amazon_val"}
            split={distilData.meta?.split || "validation"}
            n={50000}
            sourceFiles={distilData.meta?.source_files}
          >
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={weightRuns.map((w) => ({
                    weight: `${w.weight}x`,
                    val_f1_pct: Number((w.validation_macro_f1 * 100).toFixed(2)),
                    neg_recall_pct: Number((w.negative_recall * 100).toFixed(2)),
                    train_n: w.train_n,
                  }))}
                  margin={{ top: 20, right: 25, left: 0, bottom: 10 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="weight" tick={{ fill: "#cbd5e1", fontSize: 11 }} />
                  <YAxis domain={[90, 96]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Tooltip
                    formatter={(val, name) => [`${val}%`, name]}
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                  />
                  <Legend />
                  <Line
                    type="monotone"
                    dataKey="val_f1_pct"
                    name="Val Macro-F1 (%)"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ fill: "#6366f1", r: 3 }}
                  >
                    <LabelList
                      dataKey="val_f1_pct"
                      position="top"
                      formatter={(v) => `${v}%`}
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
                    />
                  </Line>
                  <Line
                    type="monotone"
                    dataKey="neg_recall_pct"
                    name="Neg Recall (%)"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: "#10b981", r: 3 }}
                  >
                    <LabelList
                      dataKey="neg_recall_pct"
                      position="bottom"
                      formatter={(v) => `${v}%`}
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
                    />
                  </Line>
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="mt-2 text-[11px] font-mono text-slate-400">
              Evaluated strictly on validation split (Rule R5). Every sweep trial used train n = {weightRuns[0]?.train_n?.toLocaleString()}.
            </div>
          </ChartCard>
        </div>

        {/* 3x Post-Hoc Test Evaluation Diagnostic Banner */}
        {testDiag.train_n && (
          <div className="rounded-2xl border border-amber-500/20 bg-amber-950/20 p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs font-mono">
            <div>
              <div className="flex items-center gap-2 font-bold text-amber-300">
                <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                <span>Diagnostic: {testDiag.display_name}</span>
              </div>
              <p className="text-[11px] text-slate-300 mt-1 font-sans">
                {testDiag.role}: train n = {testDiag.train_n?.toLocaleString()}, weight = {testDiag.weight}x, evaluated once on {testDiag.eval_set_id}.
              </p>
            </div>
            <div className="flex items-center gap-3 shrink-0 text-amber-200">
              <span>Accuracy: {(testDiag.accuracy * 100).toFixed(2)}%</span>
              <span>Macro-F1: {(testDiag.macro_f1 * 100).toFixed(2)}%</span>
              <span>Neg Recall: {(testDiag.negative_recall * 100).toFixed(2)}%</span>
            </div>
          </div>
        )}
      </div>

      {/* 3. Training Trajectory Curve & Reliability Calibration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Training Curve with connectNulls */}
        <ChartCard
          title="DistilBERT 281k Training &amp; Validation Loss"
          subtitle="Hugging Face trainer state logging steps with connectNulls handling asynchronous eval checkpoints"
          model="distilbert-base-uncased"
          evalSetId="amazon_train"
          split={trainCurve.data?.meta?.split || "validation"}
          n={281271}
          sourceFiles={trainCurve.data?.meta?.source_files}
        >
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={trainHistory}
                margin={{ top: 15, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="step" tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                <YAxis domain={[0.05, 0.3]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  formatter={(val, name) => [
                    val !== null ? Number(val).toFixed(4) : "None",
                    name,
                  ]}
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_loss"
                  name="Training Loss"
                  stroke="#6366f1"
                  strokeWidth={2}
                  dot={false}
                  connectNulls={true}
                />
                <Line
                  type="monotone"
                  dataKey="eval_loss"
                  name="Validation Loss"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={{ fill: "#f59e0b", r: 4 }}
                  connectNulls={true}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Reliability Diagram with Temperature Slider */}
        <ChartCard
          title="Reliability Calibration Diagram"
          subtitle="Precomputed temperature grids. Slider selects precomputed bin sets; does not rescale in browser."
          model={activeCalibModel.model_id || "multimodel_calibration"}
          evalSetId="amazon_test"
          split="test"
          n={72846}
          sourceFiles={calib.data?.meta?.source_files}
        >
          {/* Model Selector & Temperature Controls */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-1 font-mono text-xs">
              <button
                onClick={() => {
                  setSelectedCalibModel("distilbert");
                  setCalibT(null);
                }}
                className={`px-2.5 py-1 rounded border transition-colors ${
                  selectedCalibModel === "distilbert"
                    ? "bg-indigo-600 border-indigo-500 text-white font-bold"
                    : "border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                DistilBERT 281k
              </button>
              <button
                onClick={() => {
                  setSelectedCalibModel("deberta");
                  setCalibT(null);
                }}
                className={`px-2.5 py-1 rounded border transition-colors ${
                  selectedCalibModel === "deberta"
                    ? "bg-indigo-600 border-indigo-500 text-white font-bold"
                    : "border-slate-800 text-slate-400 hover:text-slate-200"
                }`}
              >
                DeBERTa-v3-Omni
              </button>
            </div>

            <div className="flex items-center gap-2 font-mono text-xs text-slate-300">
              <span>T = {currentTKey}</span>
              <span className="text-[10px] text-slate-500">
                (Fitted T: {activeCalibModel.fitted_T})
              </span>
            </div>
          </div>

          {/* Slider for T (0.5 to 3.0, step 0.1) */}
          <div className="mb-3 px-1">
            <input
              type="range"
              min="0.5"
              max="3.0"
              step="0.1"
              value={currentTKey}
              onChange={(e) => setCalibT(Number(e.target.value).toFixed(1))}
              className="w-full accent-indigo-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[10px] font-mono text-slate-500 mt-1">
              <span>0.5 (Underconfident)</span>
              <span>Fitted T = {activeCalibModel.fitted_T}</span>
              <span>3.0 (Over-damped)</span>
            </div>
          </div>

          {/* ECE Metric Display */}
          <div className="grid grid-cols-3 gap-2 mb-3 text-xs font-mono text-center">
            <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-[10px] text-slate-400">Raw ECE</div>
              <div className="font-bold text-slate-200">{activeCalibModel.raw_ece}</div>
            </div>
            <div className="p-2 rounded-lg bg-indigo-950/40 border border-indigo-500/30">
              <div className="text-[10px] text-indigo-300">Fitted ECE</div>
              <div className="font-bold text-indigo-300">{activeCalibModel.calibrated_ece}</div>
            </div>
            <div className="p-2 rounded-lg bg-slate-950/60 border border-slate-800">
              <div className="text-[10px] text-slate-400">Chosen T ECE</div>
              <div className="font-bold text-amber-300">{currentGrid.ece ?? activeCalibModel.calibrated_ece}</div>
            </div>
          </div>

          {/* Reliability Chart */}
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={reliabilityBins.map((b) => ({
                  bin: b.bin_label,
                  mean_conf: Number((b.mean_confidence * 100).toFixed(1)),
                  empirical_acc: Number((b.empirical_accuracy * 100).toFixed(1)),
                  count: b.count,
                }))}
                margin={{ top: 15, right: 20, left: 0, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="bin" tick={{ fill: "#cbd5e1", fontSize: 9 }} />
                <YAxis domain={[40, 100]} tick={{ fill: "#94a3b8", fontSize: 10 }} />
                <Tooltip
                  formatter={(val, name) => [`${val}%`, name]}
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="empirical_acc"
                  name="Empirical Accuracy (%)"
                  stroke="#10b981"
                  strokeWidth={2}
                  dot={{ fill: "#10b981", r: 3 }}
                >
                  <LabelList
                    dataKey="empirical_acc"
                    position="top"
                    formatter={(v) => `${v}%`}
                    style={{ fill: "#cbd5e1", fontSize: 8, fontFamily: "monospace" }}
                  />
                </Line>
                <Line
                  type="monotone"
                  dataKey="mean_conf"
                  name="Mean Confidence (%)"
                  stroke="#6366f1"
                  strokeWidth={2}
                  strokeDasharray="4 4"
                  dot={{ fill: "#6366f1", r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>
    </div>
  );
}
