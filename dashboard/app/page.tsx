import type { Metadata } from "next";
import { ResearchDashboard } from "./research-dashboard";

export const metadata: Metadata = {
  title: "GEcko — Grand Exchange Transformation Research",
  description:
    "An evidence-first research dashboard for GEcko's walk-forward statistical arbitrage study of Old School RuneScape Grand Exchange transformations.",
};

export default function Home() {
  return <ResearchDashboard />;
}
