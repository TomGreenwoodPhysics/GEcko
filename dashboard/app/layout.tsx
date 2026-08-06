import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://gecko-research-dashboard.greenwoodtom.chatgpt.site"),
  title: {
    default: "GEcko — Grand Exchange Research",
    template: "%s | GEcko",
  },
  description:
    "Walk-forward cointegration research across 11 years of Old School RuneScape Grand Exchange data.",
  openGraph: {
    title: "GEcko — Grand Exchange Research",
    description:
      "An evidence-first walk-forward statistical arbitrage study across 11 years of Grand Exchange data.",
    url: "https://gecko-research-dashboard.greenwoodtom.chatgpt.site",
    siteName: "GEcko Research",
    images: [
      {
        url: "https://gecko-research-dashboard.greenwoodtom.chatgpt.site/og-v2.png",
        width: 1731,
        height: 909,
        alt: "GEcko Grand Exchange transformation research with two clean time-series lines on a neutral grid",
      },
    ],
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "GEcko — Grand Exchange Research",
    description:
      "Walk-forward statistical arbitrage research with real costs, capacity, and lookahead auditing.",
    images: ["https://gecko-research-dashboard.greenwoodtom.chatgpt.site/og-v2.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
