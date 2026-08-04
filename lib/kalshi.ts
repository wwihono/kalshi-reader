/**
 * Minimal server-side client + helpers for the public Kalshi Trade API.
 *
 * Kalshi's market-data (read) endpoints are public and require no auth, which is
 * all "kalshi-reader" needs. The base URL can be overridden with KALSHI_API_BASE
 * (e.g. to point at the demo environment).
 */

export const KALSHI_API_BASE =
  process.env.KALSHI_API_BASE ?? "https://api.elections.kalshi.com/trade-api/v2";

export interface KalshiMarket {
  ticker: string;
  title: string;
  yes_sub_title?: string;
  no_sub_title?: string;
  status?: string;
  close_time?: string;
  yes_bid_dollars?: string;
  yes_ask_dollars?: string;
  no_bid_dollars?: string;
  no_ask_dollars?: string;
  last_price_dollars?: string;
  volume_fp?: number;
  volume_24h_fp?: number;
  open_interest_fp?: number;
  rules_primary?: string;
}

export interface KalshiEvent {
  event_ticker: string;
  series_ticker?: string;
  title: string;
  sub_title?: string;
  category?: string;
  markets: KalshiMarket[];
}

interface EventsResponse {
  events: KalshiEvent[];
  cursor?: string;
}

/**
 * Convert a Kalshi dollar-string (e.g. "0.0900") into an implied probability
 * percentage string (e.g. "9%"). Yes prices range 0..1 dollars per contract,
 * which maps directly to an implied 0..100% probability.
 */
export function toProbabilityPercent(dollars?: string | null): string {
  if (dollars === undefined || dollars === null || dollars === "") return "—";
  const value = Number(dollars);
  if (Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}%`;
}

/** Format a Kalshi dollar-string as a cents price (e.g. "0.0900" -> "9¢"). */
export function toCents(dollars?: string | null): string {
  if (dollars === undefined || dollars === null || dollars === "") return "—";
  const value = Number(dollars);
  if (Number.isNaN(value)) return "—";
  return `${Math.round(value * 100)}¢`;
}

/** Format a floating-point volume into a compact human-readable string. */
export function formatVolume(volume?: number | null): string {
  if (volume === undefined || volume === null || Number.isNaN(volume)) return "—";
  const v = Math.round(volume);
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}K`;
  return `${v}`;
}

/** Case-insensitive filter over an event's title, subtitle and category. */
export function filterEvents(events: KalshiEvent[], query: string): KalshiEvent[] {
  const q = query.trim().toLowerCase();
  if (!q) return events;
  return events.filter((event) => {
    const haystack = [event.title, event.sub_title, event.category]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return haystack.includes(q);
  });
}

async function kalshiFetch<T>(path: string): Promise<T> {
  const url = `${KALSHI_API_BASE}${path}`;
  const res = await fetch(url, {
    headers: { Accept: "application/json" },
    // Market data changes constantly; keep it fresh but allow short caching.
    next: { revalidate: 15 },
  });
  if (!res.ok) {
    throw new Error(`Kalshi API ${res.status} for ${path}`);
  }
  return (await res.json()) as T;
}

/** Fetch a page of open events with their nested markets. */
export async function getOpenEvents(limit = 60): Promise<KalshiEvent[]> {
  const data = await kalshiFetch<EventsResponse>(
    `/events?limit=${limit}&status=open&with_nested_markets=true`,
  );
  return data.events ?? [];
}

/** Fetch a single event (with nested markets) by its ticker. */
export async function getEvent(eventTicker: string): Promise<KalshiEvent | null> {
  const data = await kalshiFetch<{ event: KalshiEvent }>(
    `/events/${encodeURIComponent(eventTicker)}?with_nested_markets=true`,
  );
  return data.event ?? null;
}
