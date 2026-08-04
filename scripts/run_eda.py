"""Run Exploratory Data Analysis and write summary JSON, figures, and eda.pdf."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kalshi_nba.eda.report_data import build_eda_payload, save_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing kalshi/ and basketball_reference/ raw files",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("reports/figures"),
        help="Directory for EDA figure PNGs",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=Path("data/processed/eda_summary.json"),
        help="Path for machine-readable EDA summary JSON",
    )
    parser.add_argument(
        "--pdf-out",
        type=Path,
        default=Path("reports/eda.pdf"),
        help="Path for eda.pdf report",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF generation",
    )
    args = parser.parse_args()

    payload = build_eda_payload(args.raw_dir, args.figures_dir)
    save_payload(payload, args.summary_out)
    print(json.dumps({
        "kalshi_rows": payload["kalshi"]["shape"]["n_rows"],
        "br_rows": payload["basketball_reference"]["shape"]["n_rows"],
        "joined_rows": payload["joined"]["shape"]["n_rows"],
        "summary": str(args.summary_out),
        "figures": payload["figures"],
    }, indent=2))

    if not args.skip_pdf:
        import importlib.util

        pdf_module_path = Path(__file__).resolve().parent / "generate_eda_pdf.py"
        spec = importlib.util.spec_from_file_location("generate_eda_pdf", pdf_module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load {pdf_module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pdf_path = module.write_pdf(args.summary_out, args.pdf_out)
        print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
