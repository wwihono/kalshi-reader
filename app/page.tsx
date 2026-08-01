import EventBrowser from "./EventBrowser";
import { getOpenEvents } from "@/lib/kalshi";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  let events;
  try {
    events = await getOpenEvents(60);
  } catch (err) {
    return (
      <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-6 text-sm">
        <p className="font-medium text-red-300">
          Could not load markets from Kalshi.
        </p>
        <p className="mt-1 text-red-200/80">
          {err instanceof Error ? err.message : "Unknown error"}
        </p>
      </div>
    );
  }

  return <EventBrowser events={events} />;
}
