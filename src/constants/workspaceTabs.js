import {
  Database,
  Trophy,
  BarChart3,
  Cpu,
  Globe,
  Bot,
  Bug,
} from "lucide-react";

export const WORKSPACE_TABS = [
  { id: 0, code: "01", label: "Executive Overview", icon: Database },
  { id: 1, code: "02", label: "Champion: DeBERTa-v3-Omni", icon: Trophy },
  { id: 2, code: "03", label: "Benchmarks and Model Archive", icon: BarChart3 },
  { id: 3, code: "04", label: "Inference Lab", icon: Cpu },
  { id: 4, code: "05", label: "Cross-Domain Analytics", icon: Globe },
  { id: 5, code: "06", label: "AI Review Analyst", icon: Bot },
  { id: 6, code: "07", label: "Error Taxonomy and Diagnostics", icon: Bug },
];
