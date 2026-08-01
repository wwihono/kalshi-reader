"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import {
  filterEvents,
  formatVolume,
  toCents,
  toProbabilityPercent,
  type KalshiEvent,
} from "@/lib/kalshi";

function eventVolume(event: KalshiEvent): number {
  return event.markets.reduce((sum, m) => sum + (m.volume_fp ?? 0), 0);
}

export default function EventBrowser({ events }: { events: KalshiEvent[] }) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => filterEvents(events, query), [events, query]);

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Live prediction markets
        </h1>
        <p className="text-sm text-slate-400">
          Browsing {events.length} open events from Kalshi. Prices show the
          implied probability of the leading outcome.
        </p>
      </div>

      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search markets by keyword (e.g. Pope, weather, election)…"
        aria-label="Search markets"
        className="w-full rounded-lg border border-white/10 bg-panel px-4 py-3 text-sm outline-none placeholder:text-slate-500 focus:border-accent"
      />

      <p className="text-xs text-slate-500" data-testid="result-count">
        {filtered.length} result{filtered.length === 1 ? "" : "s"}
      </p>

      <ul className="grid gap-3">
        {filtered.map((event) => {
          const top = event.markets[0];
          return (
            <li key={event.event_ticker}>
              <Link
                href={`/event/${encodeURIComponent(event.event_ticker)}`}
                className="block rounded-xl border border-white/10 bg-panel p-4 transition hover:border-accent/60 hover:bg-white/5"
              >
                <div className="flex items-start justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      {event.category && (
                        <span className="rounded-full bg-white/10 px-2 py-0.5 text-[11px] uppercase tracking-wide text-slate-300">
                          {event.category}
                        </span>
                      )}
                      <span className="text-xs text-slate-500">
                        {event.markets.length} market
                        {event.markets.length === 1 ? "" : "s"}
                      </span>
                    </div>
                    <h2 className="text-base font-medium leading-snug">
                      {event.title}
                    </h2>
                    {event.sub_title && (
                      <p className="text-sm text-slate-400">{event.sub_title}</p>
                    )}
                  </div>
                  {top && (
                    <div className="shrink-0 text-right">
                      <div className="text-2xl font-semibold text-accent">
                        {toProbabilityPercent(top.yes_bid_dollars)}
                      </div>
                      <div className="text-xs text-slate-500">
                        vol {formatVolume(eventVolume(event))}
                      </div>
                    </div>
                  )}
                </div>
                {event.markets.length > 1 && top && (
                  <p className="mt-3 text-xs text-slate-500">
                    Leading:{" "}
                    <span className="text-slate-300">
                      {top.yes_sub_title || top.title}
                    </span>{" "}
                    at {toCents(top.yes_ask_dollars)}
                  </p>
                )}
              </Link>
            </li>
          );
        })}
      </ul>

      {filtered.length === 0 && (
        <p className="rounded-lg border border-white/10 bg-panel p-6 text-center text-sm text-slate-400">
          No markets match “{query}”.
        </p>
      )}
    </div>
  );
}
