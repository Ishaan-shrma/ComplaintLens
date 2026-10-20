import React from "react";
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
  AreaChart,
  Area,
} from "recharts";
import {
  Database,
  Filter,
  Trash2,
  CheckCircle,
  Scissors,
  Split,
  Globe,
  Info,
} from "lucide-react";

export default function Tab1Corpus() {
  const { loading, missing, missingPath, data } = useData("corpus.json");

  if (loading) {
    return (
      <div className="p-8 font-mono text-xs text-slate-400">
        Loading corpus.json ...
      </div>
    );
  }

  if (missing || !data) {
    return <DataMissingPanel path={missingPath} />;
  }

  const meta = data.meta || {};
  const evalSet = meta.eval_set || {};
  const perPlatform = data.per_platform || [];
  const yearlyVolume = data.yearly_volume || [];

  // Funnel breakdown for visualization
  const funnelSteps = [
    { name: "Raw Ingested", count: data.raw, fill: "#6366f1" },
    { name: "Clean Corpus", count: data.clean, fill: "#3b82f6" },
    { name: "Binary Pool", count: data.binary_pool, fill: "#06b6d4" },
    { name: "Train Split", count: data.train, fill: "#10b981" },
    { name: "Held-Out Test", count: data.test, fill: "#a855f7" },
    { name: "Validation", count: data.validation, fill: "#f59e0b" },
  ];

  // Per platform chart data
  const platformChartData = perPlatform.map((p) => ({
    name: p.platform,
    domain: p.domain,
    n: p.n,
    positive_pct: p.binary_class_split?.positive_pct,
    negative_pct: p.binary_class_split?.negative_pct,
    n_negative: p.n_negative,
    n_positive: p.n_positive,
  }));

  const starChartData = data.star_counts
    ? Object.keys(data.star_counts)
        .sort()
        .map((star) => ({
          rating: `${star} Star`,
          count: data.star_counts[star],
        }))
    : [];

  return (
    <div className="space-y-6">
      {/* Workspace Header */}
      <div className="border-b border-slate-800 pb-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="font-mono text-xs uppercase tracking-wider text-indigo-400 font-bold">
            [01] Executive Overview
          </span>
          <span className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] font-mono text-slate-400">
            corpus.json
          </span>
        </div>
        <h2 className="text-xl md:text-2xl font-bold tracking-tight text-slate-100 font-mono">
          Corpus Architecture, Data Cleaning &amp; Split Partitions
        </h2>
        <p className="text-xs text-slate-400 mt-1">
          KPI metrics covering raw ingest, deduplication, invalid record purging, neutral filtering, and partition sizes.
        </p>
      </div>

      {/* KPI Cards Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
        <MetricCard
          label="Raw Records"
          value={data.raw?.toLocaleString()}
          sublabel="Initial raw ingest"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.raw}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Duplicates Removed"
          value={data.duplicates_removed?.toLocaleString()}
          sublabel="Exact hash & text duplicates"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.duplicates_removed}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Invalid Removed"
          value={data.invalid_removed?.toLocaleString()}
          sublabel="Malformed or null rows"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.invalid_removed}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Clean Corpus"
          value={data.clean?.toLocaleString()}
          sublabel="Valid unique reviews"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.clean}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Neutral Excluded"
          value={data.neutral_excluded?.toLocaleString()}
          sublabel="3-star ambiguous reviews"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.neutral_excluded}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Train / Val / Test"
          value={`${data.train?.toLocaleString()} / ${data.validation?.toLocaleString()} / ${data.test?.toLocaleString()}`}
          sublabel="Partition allocations"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.binary_pool}
          pathForMissing={missingPath}
        />
        <MetricCard
          label="Omni Total (4 Platforms)"
          value={evalSet.n?.toLocaleString()}
          sublabel="Multi-domain aggregate"
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
          pathForMissing={missingPath}
        />
      </div>

      {/* Dynamic Data Take-away Banner with Source File Citation */}
      <div className="rounded-2xl border border-indigo-500/20 bg-indigo-950/20 p-4 font-mono text-xs text-indigo-300 space-y-2">
        <div className="flex items-center gap-2 font-bold text-indigo-200">
          <Info className="w-4 h-4 text-indigo-400 shrink-0" />
          <span>Corpus Take-Away &amp; Class Distribution Clarification</span>
        </div>
        <p className="text-[11px] leading-relaxed text-slate-300 font-sans">
          Deduplication purged {data.duplicates_removed?.toLocaleString()} duplicate entries and {data.invalid_removed} malformed records from the initial {data.raw?.toLocaleString()} raw ingested reviews, yielding {data.clean?.toLocaleString()} clean reviews.
          Neutral reviews ({data.neutral_excluded?.toLocaleString()} items) were isolated from the binary pool to prevent ambiguous gradient updates.
        </p>
        <p className="text-[11px] leading-relaxed text-amber-300/90 font-sans">
          Class Split Context: The {perPlatform[0]?.binary_class_split?.positive_pct}% positive vs {perPlatform[0]?.binary_class_split?.negative_pct}% negative ratio applies specifically to the {perPlatform[0]?.platform || "Amazon"} binary split ({data.binary_pool?.toLocaleString()} reviews), not to the total multi-platform corpus. As detailed below, each consumer platform maintains distinct positive and negative class proportions.
        </p>
        <div className="text-[10px] text-slate-500 font-mono pt-1 border-t border-indigo-500/10">
          Source Files: {meta.source_files?.join(", ") || "data/processed/reviews_cleaned.csv, data/processed/omni_corpus.csv"}
        </div>
      </div>

      {/* Two Column Visual Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Engineering Funnel Chart */}
        <ChartCard
          title="Data Ingestion &amp; Partition Funnel"
          subtitle="Stage-by-stage sample preservation from raw ingestion to held-out test set"
          model="Corpus Pipeline"
          evalSetId={evalSet.id}
          split={meta.split}
          n={data.raw}
          sourceFiles={meta.source_files}
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={funnelSteps}
                layout="vertical"
                margin={{ top: 10, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis type="number" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fill: "#cbd5e1", fontSize: 11 }}
                  width={110}
                />
                <Tooltip
                  formatter={(val) => [typeof val === "number" ? val.toLocaleString() : val, "Records"]}
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                />
                <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                  <LabelList
                    dataKey="count"
                    position="right"
                    formatter={(v) => (typeof v === "number" ? v.toLocaleString() : v)}
                    style={{ fill: "#94a3b8", fontSize: 10, fontFamily: "monospace" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* Per-Platform Corpus Breakdown */}
        <ChartCard
          title="Per-Platform Corpus Sizes &amp; Class Mix"
          subtitle="Sample volume and positive vs negative proportions across all four platforms"
          model="Omni Corpus"
          evalSetId={evalSet.id}
          split={meta.split}
          n={evalSet.n}
          sourceFiles={meta.source_files}
        >
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={platformChartData}
                margin={{ top: 20, right: 20, left: 10, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis dataKey="name" tick={{ fill: "#cbd5e1", fontSize: 10 }} />
                <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
                <Tooltip
                  formatter={(val, name) => [
                    typeof val === "number" ? val.toLocaleString() : val,
                    name === "n" ? "Total Reviews" : name,
                  ]}
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }}
                />
                <Legend />
                <Bar
                  dataKey="n"
                  name="Platform n"
                  fill="#6366f1"
                  radius={[4, 4, 0, 0]}
                >
                  <LabelList
                    dataKey="n"
                    position="top"
                    formatter={(v) => (typeof v === "number" ? v.toLocaleString() : v)}
                    style={{ fill: "#cbd5e1", fontSize: 10, fontFamily: "monospace" }}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Per-Platform Class Split Table */}
      <div className="rounded-2xl border border-slate-800 bg-slate-900 p-5">
        <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-2">
          <h3 className="text-sm font-bold text-slate-100 font-mono">
            Platform-Specific Class Stratification
          </h3>
          <span className="text-[10px] font-mono text-slate-500">
            Evaluated per individual platform corpus
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 px-3">Platform</th>
                <th className="py-2 px-3">Domain</th>
                <th className="py-2 px-3">Total n</th>
                <th className="py-2 px-3">Negative Count</th>
                <th className="py-2 px-3">Positive Count</th>
                <th className="py-2 px-3">Positive %</th>
                <th className="py-2 px-3">Negative %</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {perPlatform.map((p) => (
                <tr key={p.platform} className="hover:bg-slate-800/40">
                  <td className="py-2.5 px-3 font-semibold text-slate-200">
                    {p.platform}
                  </td>
                  <td className="py-2.5 px-3 text-slate-400">{p.domain}</td>
                  <td className="py-2.5 px-3">{p.n?.toLocaleString()}</td>
                  <td className="py-2.5 px-3 text-rose-400">
                    {p.n_negative?.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 text-emerald-400">
                    {p.n_positive?.toLocaleString()}
                  </td>
                  <td className="py-2.5 px-3 text-emerald-400">
                    {p.binary_class_split?.positive_pct}%
                  </td>
                  <td className="py-2.5 px-3 text-rose-400">
                    {p.binary_class_split?.negative_pct}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
