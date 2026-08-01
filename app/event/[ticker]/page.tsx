import Link from "next/link";
import { notFound } from "next/navigation";
import {
  formatVolume,
  getEvent,
  toCents,
  toProbabilityPercent,
} from "@/lib/kalshi";

export const dynamic = "force-dynamic";

export default async function EventPage({
  params,
}: {
  params: { ticker: string };
}) {
  const ticker = decodeURIComponent(params.ticker);
  const event = await getEvent(ticker);
  if (!event) notFound();

  return (
    <div className="space-y-6">
      <Link
        href="/"
        className="inline-flex items-center gap-1 text-sm text-slate-400 hover:text-slate-200"
      >
        ← All markets
      </Link>

      <div className="space-y-2">
        {event.category && (
          <span className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] uppercase tracking-wide text-slate-300">
            {event.category}
          </span>
        )}
        <h1 className="text-2xl font-semibold tracking-tight">{event.title}</h1>
        {event.sub_title && (
          <p className="text-slate-400">{event.sub_title}</p>
        )}
        <p className="text-xs text-slate-500">{event.event_ticker}</p>
      </div>

      <div className="overflow-hidden rounded-xl border border-white/10">
        <table className="w-full text-sm">
          <thead className="bg-white/5 text-left text-xs uppercase tracking-wide text-slate-400">
            <tr>
              <th className="px-4 py-3">Outcome</th>
              <th className="px-4 py-3 text-right">Yes (buy)</th>
              <th className="px-4 py-3 text-right">No (buy)</th>
              <th className="px-4 py-3 text-right">Implied</th>
              <th className="px-4 py-3 text-right">Volume</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {event.markets.map((market) => (
              <tr key={market.ticker} className="hover:bg-white/5">
                <td className="px-4 py-3">
                  <div className="font-medium">
                    {market.yes_sub_title || market.title}
                  </div>
                  <div className="text-xs text-slate-500">{market.ticker}</div>
                </td>
                <td className="px-4 py-3 text-right text-accent">
                  {toCents(market.yes_ask_dollars)}
                </td>
                <td className="px-4 py-3 text-right text-slate-300">
                  {toCents(market.no_ask_dollars)}
                </td>
                <td className="px-4 py-3 text-right font-semibold">
                  {toProbabilityPercent(market.yes_bid_dollars)}
                </td>
                <td className="px-4 py-3 text-right text-slate-400">
                  {formatVolume(market.volume_fp)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {event.markets[0]?.rules_primary && (
        <div className="rounded-xl border border-white/10 bg-panel p-4">
          <h2 className="mb-2 text-sm font-semibold text-slate-300">Rules</h2>
          <p className="text-sm leading-relaxed text-slate-400">
            {event.markets[0].rules_primary}
          </p>
        </div>
      )}
    </div>
  );
}
