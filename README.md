# kalshi-reader

A small web app for browsing **live [Kalshi](https://kalshi.com) prediction-market data**.
Built with Next.js (App Router), TypeScript and Tailwind CSS. It reads the public
Kalshi Trade API (no authentication required) server-side, so there are no secrets
to configure for local development.

## Features

- Browse open Kalshi events with their nested markets
- Client-side keyword search/filter across title, subtitle and category
- Event detail page with per-outcome Yes/No prices, implied probability and volume

## Requirements

- Node.js 22+
- pnpm 10+

## Getting started

```bash
pnpm install        # install dependencies
pnpm dev            # start the dev server at http://localhost:3000
```

Other scripts:

```bash
pnpm lint           # run ESLint (next lint)
pnpm test           # run unit tests (vitest)
pnpm build          # production build
pnpm start          # run the production build
```

## Configuration

By default the app reads Kalshi's public production data host. To point it
elsewhere (e.g. the demo environment), set:

```bash
KALSHI_API_BASE="https://demo-api.kalshi.co/trade-api/v2" pnpm dev
```

## Project layout

- `app/` — Next.js App Router pages (`page.tsx` markets list, `event/[ticker]` detail)
- `app/EventBrowser.tsx` — client component providing the search/filter UI
- `lib/kalshi.ts` — public Kalshi API client + pure formatting/filter helpers
- `lib/kalshi.test.ts` — unit tests for the pure helpers
