"""HTTP GET with retries for failures that are worth trying again."""
import random
import time

import httpx

RETRY_STATUSES = {429, 500, 502, 503, 504}  # rate limited or server trouble: may succeed later
MAX_ATTEMPTS = 3
BASE_DELAY = 1.0   # seconds; doubles on each retry (1s, 2s, 4s...)
MAX_DELAY = 20.0
_sleep = time.sleep  # tests swap this out so they never actually wait


def _delay(attempt: int, response: httpx.Response | None) -> float:
    retry_after = response.headers.get("retry-after", "") if response is not None else ""
    if retry_after.isdigit():  # the server told us how long to wait
        return min(float(retry_after), MAX_DELAY)
    backoff = BASE_DELAY * 2 ** (attempt - 1)
    return min(backoff, MAX_DELAY) + random.uniform(0, 0.5)  # jitter avoids synchronized retries


def get_with_retry(url: str, *, attempts: int = MAX_ATTEMPTS, **kwargs) -> httpx.Response:
    """httpx.get that retries timeouts, connection errors, 429 and 5xx with exponential backoff.

    4xx errors like 404 are NOT retried: asking again won't change the answer.
    Returns the last response (which may still be a 429/5xx after all attempts),
    or re-raises the last network error.
    """
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.get(url, **kwargs)
        except httpx.TransportError:  # timeouts, DNS failures, refused connections
            if attempt == attempts:
                raise
            _sleep(_delay(attempt, None))
            continue
        if response.status_code not in RETRY_STATUSES or attempt == attempts:
            return response
        _sleep(_delay(attempt, response))
    raise AssertionError("unreachable")
