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
import { Globe, Layers, BookOpen, Tag, Sparkles, CheckCircle2 } from "lucide-react";

export default function Tab5CrossDomain() {
  const cross = useData("omni_cross_domain_metrics.json");
  const vocab = useData("domain_vocab.json");
  const [selectedPlatform, setSelectedPlatform] = useState("Amazon Fine Food");

  if (cross.loading || vocab.loading) {
    return (
      <div className="p-8 font-mono text-xs text-slate-400">
        Loading cross-domain analytics &amp; domain vocabulary ...
      </div>
    );
  }

  if (cross.missing || !cross.data) {
    return <DataMissingPanel path={cross.missingPath} />;
  }

  const crossData = cross.data;
  const meta = crossData.meta || {};
  const evalSet = meta.eval_set || {};
  const platforms = crossData.platforms || [];
  const vocabData = vocab.data?.platforms || {};

  // Find active platform details
  const activePlat =
    platforms.find((p) => p.platform === selectedPlatform) || platforms[0] || {};
  const activeVocab = vocabData[activePlat.platform] || [];

  // Chart data for comparing cross-domain performance
  const perfChartData = platforms.map((p) => ({
    name: p.platform,
    domain: p.domain,
    macro_f1_pct: Number((p.macro_f1 * 100).toFixed(2)),
    negative_recall_pct: Number((p.negative_recall * 100).toFixed(2)),
    accuracy_pct: Number((p.accuracy * 100).toFixed(2)),
    in_training: p.in_training_mix,
    n: p.n,
  }));

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-xs uppercase tracking-wider text-indigo-400 font-bold">
            [05] Cross-Domain Analytics
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-400">
            omni_cross_domain_metrics.json • domain_vocab.json
          </span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          Multi-Platform Performance &amp; Domain Lexicon Analytics
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          Comparative multi-platform evaluation across Amazon Fine Food, Flipkart Retail, Yelp Dining, and IMDb Cinema under a multi-platform replay buffer blend.
        </p>
      </div>

      {/* KPI Cards across the 4 Platforms */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {platforms.map((p) => (
          <div
            key={p.platform}
            onClick={() => setSelectedPlatform(p.platform)}
            className={`rounded-2xl border p-4 cursor-pointer transition-all ${
              selectedPlatform === p.platform
                ? "border-indigo-500 bg-indigo-950/30 shadow-[0_0_20px_rgba(99,102,241,0.15)]"
                : "border-slate-800 bg-slate-900 hover:border-slate-700"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-slate-200">
                {p.platform}
              </span>
              {p.in_training_mix ? (
                <span className="px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 text-[9px] font-mono uppercase">
                  {p.training_role || "Replay Buffer"}
                </span>
              ) : (
                <span className="px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 text-[9px] font-mono uppercase">
                  Held-Out Unseen
                </span>
              )}
            </div>
            <div className="text-lg font-bold text-slate-100 font-mono">
              {(p.macro_f1 * 100).toFixed(2)}%
            </div>
            <div className="text-[10px] font-mono text-slate-400 mt-1">
              Macro-F1 • Neg Recall: {(p.negative_recall * 100).toFixed(2)}%
            </div>
            <div className="text-[10px] font-mono text-slate-500 mt-2 pt-2 border-t border-slate-800">
              n = {p.n?.toLocaleString()} ({p.domain})
            </div>
          </div>
        ))}
      </div>

      {/* Two Column Layout: Cross-Platform Chart vs Domain Vocab */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Performance Comparison Chart */}
        <div className="lg:col-span-7">
          <ChartCard
            title="Multi-Platform Macro-F1 &amp; Negative Recall"
            subtitle={crossData.evaluation_notes?.training_blend ? `Cross-domain performance across 4 platforms fine-tuned with ${crossData.evaluation_notes.training_blend}` : "Cross-domain performance across 4 platforms fine-tuned with Omni replay buffer"}
            model={meta.model_id}
            evalSetId={evalSet.id}
            split={meta.split}
            n={evalSet.n}
            sourceFiles={meta.source_files}
          >
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={perfChartData}
                  margin={{ top: 20, right: 20, left: 0, bottom: 20 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                  <XAxis
                    dataKey="name"
                    tick={{ fill: "#cbd5e1", fontSize: 10 }}
                    interval={0}
                  />
                  <YAxis domain={[80, 100]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
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
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
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
                      style={{ fill: "#cbd5e1", fontSize: 9, fontFamily: "monospace" }}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </ChartCard>
        </div>

        {/* Right: Selected Platform Domain Detail & Vocabulary */}
        <div className="lg:col-span-5 space-y-4">
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-sm font-bold text-slate-100 font-mono">
                  {activePlat.platform}
                </h3>
                <span className="text-xs text-indigo-400 font-mono">
                  {activePlat.domain}
                </span>
              </div>
              <span className="text-xs font-mono text-slate-400">
                n = {activePlat.n?.toLocaleString()}
              </span>
            </div>

            {/* Label Derivation Note */}
            <div className="rounded-xl border border-slate-800 bg-slate-950/70 p-3 font-mono text-xs space-y-1">
              <div className="text-[10px] text-slate-400 uppercase font-bold flex items-center gap-1.5">
                <BookOpen className="w-3.5 h-3.5 text-indigo-400" />
                <span>Label Derivation Methodology</span>
              </div>
              <p className="text-slate-300 font-sans leading-relaxed text-[11px]">
                {activePlat.label_derivation}
              </p>
            </div>

            {/* Class Mix */}
            <div className="grid grid-cols-2 gap-3 text-xs font-mono">
              <div className="p-3 rounded-xl border border-emerald-500/20 bg-emerald-950/10">
                <div className="text-[10px] text-emerald-400">Positive Class</div>
                <div className="text-base font-bold text-emerald-300 mt-0.5">
                  {activePlat.class_mix?.positive_pct}%
                </div>
              </div>
              <div className="p-3 rounded-xl border border-rose-500/20 bg-rose-950/10">
                <div className="text-[10px] text-rose-400">Negative Class</div>
                <div className="text-base font-bold text-rose-300 mt-0.5">
                  {activePlat.class_mix?.negative_pct}%
                </div>
              </div>
            </div>

            {/* Top Distinguishing Terms */}
            <div className="pt-2 border-t border-slate-800">
              <div className="text-xs font-mono font-bold text-slate-200 mb-2 flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <Tag className="w-3.5 h-3.5 text-indigo-400" />
                  Top Distinguishing Lexicon Terms
                </span>
                <span className="text-[10px] text-slate-500">TF-IDF Distinctiveness</span>
              </div>

              {activeVocab.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {activeVocab.map((item, idx) => (
                    <div
                      key={idx}
                      className="px-2.5 py-1 rounded-lg border border-slate-800 bg-slate-950/80 font-mono text-xs flex items-center gap-2"
                    >
                      <span className="text-slate-200 font-bold">{item.term}</span>
                      <span className="text-[10px] text-indigo-400 px-1 py-0.2 rounded bg-indigo-500/10">
                        {item.score}
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-xs font-mono text-slate-500 py-4 text-center">
                  No vocabulary terms extracted for this platform.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Held-Out Multi-Domain Benchmark Table (DeBERTa vs DistilBERT Baseline) */}
      {crossData.heldout_multidomain_benchmark && (() => {
        const hData = crossData.heldout_multidomain_benchmark;
        const hPlatforms = hData.platforms || {};
        const hReal = hData.real_platforms_overall || hData.overall || {};
        const hOverall = hData.overall || {};
        const yelp = hPlatforms.yelp || {};
        const imdb = hPlatforms.imdb || {};
        const flip = hPlatforms.flipkart || {};

        const rows = [
          {
            name: "Yelp Dining (Hospitality, Real)",
            p: yelp,
            synthetic: false,
          },
          {
            name: "IMDb Cinema (Media, Real)",
            p: imdb,
            synthetic: false,
          },
          {
            name: "Flipkart Retail (Synthetic Template)*",
            p: flip,
            synthetic: true,
          },
        ];

        return (
          <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-sm space-y-3">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-3">
              <div>
                <div className="flex items-center gap-2">
                  <Globe className="w-4 h-4 text-indigo-400" />
                  <h3 className="text-sm font-bold text-slate-100 font-mono">
                    Held-Out Multi-Domain Benchmark (DeBERTa vs DistilBERT Baseline)
                  </h3>
                </div>
                <p className="text-xs text-slate-400 mt-0.5">
                  Evaluated on unseen binary reviews with zero training overlap across held-out rows from multi-domain platforms.
                </p>
              </div>
              <span className="px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/30 text-[10px] font-mono text-indigo-300 font-bold shrink-0">
                {hData.evaluation_file}
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs font-mono">
                <thead>
                  <tr className="border-b border-slate-800 text-slate-400 text-[11px]">
                    <th className="py-2.5 px-3">Platform Split</th>
                    <th className="py-2.5 px-3">Binary n (% Neg)</th>
                    <th className="py-2.5 px-3">DistilBERT Neg Recall</th>
                    <th className="py-2.5 px-3">DeBERTa Neg Recall</th>
                    <th className="py-2.5 px-3">DistilBERT Macro-F1</th>
                    <th className="py-2.5 px-3">DeBERTa Macro-F1</th>
                    <th className="py-2.5 px-3">Missed Complaints (FP)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 text-slate-200">
                  {rows.map((r) => {
                    const d = r.p.distilbert_281k || {};
                    const b = r.p.deberta_v3_omni || {};
                    const fpDelta =
                      d.fp && b.fp !== undefined
                        ? (((b.fp - d.fp) / d.fp) * 100).toFixed(1)
                        : "0.0";
                    return (
                      <tr key={r.name} className="hover:bg-slate-800/30">
                        <td className="py-3 px-3 font-semibold text-slate-100">
                          {r.name}
                        </td>
                        <td className="py-3 px-3 text-slate-400">
                          {r.p.n?.toLocaleString()} ({r.p.negative_share_pct}%)
                        </td>
                        <td className="py-3 px-3 text-rose-300">
                          {(d.negative_recall * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 font-bold text-emerald-400">
                          {(b.negative_recall * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 text-slate-400">
                          {(d.macro_f1 * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 font-bold text-emerald-400">
                          {(b.macro_f1 * 100).toFixed(2)}%
                        </td>
                        <td className="py-3 px-3 text-emerald-400 font-semibold">
                          {d.fp} &rarr; {b.fp} ({fpDelta}%)
                        </td>
                      </tr>
                    );
                  })}

                  {/* Primary Real Platforms Headline Row */}
                  <tr className="bg-indigo-950/40 font-bold border-t-2 border-indigo-500/40">
                    <td className="py-3 px-3 text-indigo-300">
                      Real Platforms (Yelp + IMDb)
                    </td>
                    <td className="py-3 px-3 text-slate-300">
                      {hReal.n?.toLocaleString()} ({hReal.negative_share_pct}%)
                    </td>
                    <td className="py-3 px-3 text-rose-300 font-bold">
                      {(hReal.distilbert_281k?.negative_recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-emerald-400 font-bold">
                      {(hReal.deberta_v3_omni?.negative_recall * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-slate-300">
                      {(hReal.distilbert_281k?.macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-emerald-400 font-bold">
                      {(hReal.deberta_v3_omni?.macro_f1 * 100).toFixed(2)}%
                    </td>
                    <td className="py-3 px-3 text-emerald-400 font-bold">
                      {hReal.distilbert_281k?.fp} &rarr; {hReal.deberta_v3_omni?.fp} (
                      {(
                        ((hReal.deberta_v3_omni?.fp - hReal.distilbert_281k?.fp) /
                          hReal.distilbert_281k?.fp) *
                        100
                      ).toFixed(1)}
                      %)
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
            <div className="text-[11px] font-mono text-slate-500 pt-1 space-y-1">
              <div>
                * Flipkart Disclosure: Evaluated on synthetic template fallback data generated during offline pipeline build; template saturation explains ceiling metrics.
              </div>
              <div>
                Real Multi-Domain Lift: DeBERTa achieves +{hReal.delta?.negative_recall_pp?.toFixed(2)} pp negative recall lift and {hReal.delta?.missed_complaints_reduction_pct?.toFixed(1)}% missed complaint reduction on genuine Yelp + IMDb reviews.
              </div>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
