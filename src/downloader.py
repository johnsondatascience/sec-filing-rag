"""Download 10-K filings from SEC EDGAR."""

from sec_edgar_downloader import Downloader

from src.config import RAW_DIR, SEC_COMPANY_NAME, SEC_EMAIL, TICKERS


def download_10k_filings(
    tickers: list[str] | None = None,
    limit: int = 2,
) -> dict[str, list[str]]:
    """Download recent 10-K filings for each ticker.

    Returns dict mapping ticker to list of downloaded filing directory paths.
    """
    tickers = tickers or TICKERS
    dl = Downloader(SEC_COMPANY_NAME, SEC_EMAIL, str(RAW_DIR))
    results: dict[str, list[str]] = {}

    for ticker in tickers:
        try:
            dl.get("10-K", ticker, limit=limit)
        except ValueError as e:
            print(f"  SKIP {ticker}: {e}")
            results[ticker] = []
            continue
        filing_dir = RAW_DIR / "sec-edgar-filings" / ticker / "10-K"
        if filing_dir.exists():
            results[ticker] = sorted(
                [str(d) for d in filing_dir.iterdir() if d.is_dir()]
            )
        else:
            results[ticker] = []

    return results


if __name__ == "__main__":
    results = download_10k_filings(tickers=["NFLX"], limit=1)
    for ticker, paths in results.items():
        print(f"{ticker}: {len(paths)} filings downloaded")
        for p in paths:
            print(f"  {p}")
