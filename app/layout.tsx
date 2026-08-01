import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Kalshi Reader",
  description: "Browse live Kalshi prediction markets",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="text-slate-100 antialiased">
        <header className="border-b border-white/10">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-4 py-4">
            <Link href="/" className="flex items-center gap-2">
              <span className="inline-block h-3 w-3 rounded-full bg-accent" />
              <span className="text-lg font-semibold tracking-tight">
                Kalshi Reader
              </span>
            </Link>
            <a
              href="https://kalshi.com"
              target="_blank"
              rel="noreferrer"
              className="text-sm text-slate-400 hover:text-slate-200"
            >
              Live market data
            </a>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
