# AGENTS.md

## Cursor Cloud specific instructions

`kalshi-reader` is a single Next.js (App Router) + TypeScript + Tailwind web app that
reads **live public Kalshi prediction-market data** server-side. There is only one
service. Standard commands live in `package.json` scripts and `README.md`; use those
as the source of truth. Notes below are the non-obvious bits.

- Package manager is **pnpm** (Node 22, pnpm 10). Dependencies are refreshed
  automatically by the startup update script (`pnpm install`), so you normally do
  not need to install anything manually.
- Run the dev server with `pnpm dev` (serves on `http://localhost:3000`). Lint /
  test / build are `pnpm lint`, `pnpm test` (vitest), `pnpm build`.
- **Network egress is required at request time.** Pages fetch the public Kalshi
  Trade API (`https://api.elections.kalshi.com/trade-api/v2`, no auth) inside
  server components. If outbound HTTPS to that host is blocked, the home page
  renders a red "Could not load markets from Kalshi" error instead of market
  cards — that indicates a network/egress problem, not an app bug. There are **no
  secrets or API keys** to configure.
- The Kalshi base URL is overridable via the `KALSHI_API_BASE` env var (e.g. to
  point at a demo host); default is the production elections host.
- Both pages use `export const dynamic = "force-dynamic"`, so they fetch fresh
  data on every request (with a short 15s revalidate on the fetch). Market prices
  will therefore change between page loads — that is expected.
- pnpm blocks dependency build scripts by default. `esbuild` and `sharp` are
  pre-approved via `pnpm.onlyBuiltDependencies` in `package.json`. The
  `unrs-resolver` build script stays ignored and can be left as-is; `pnpm lint`
  still passes without it.
