import httpx
import pytest
import respx

from seo_audit_agent.net import get_with_retry

URL = "https://shop.test/"


@respx.mock
def test_retries_server_error_then_succeeds(no_retry_sleep):
    route = respx.get(URL).mock(side_effect=[httpx.Response(503), httpx.Response(200, text="ok")])
    r = get_with_retry(URL)
    assert r.status_code == 200
    assert route.call_count == 2
    assert len(no_retry_sleep) == 1          # waited once between attempts


@respx.mock
def test_does_not_retry_404(no_retry_sleep):
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    assert get_with_retry(URL).status_code == 404
    assert route.call_count == 1
    assert no_retry_sleep == []


@respx.mock
def test_honours_retry_after_header(no_retry_sleep):
    respx.get(URL).mock(side_effect=[httpx.Response(429, headers={"Retry-After": "7"}),
                                     httpx.Response(200)])
    get_with_retry(URL)
    assert no_retry_sleep == [7.0]


@respx.mock
def test_gives_up_after_max_attempts_and_returns_last_response(no_retry_sleep):
    route = respx.get(URL).mock(return_value=httpx.Response(503))
    assert get_with_retry(URL, attempts=3).status_code == 503
    assert route.call_count == 3
    assert len(no_retry_sleep) == 2          # no pointless wait after the final attempt


@respx.mock
def test_network_error_is_retried_then_raised(no_retry_sleep):
    route = respx.get(URL).mock(side_effect=httpx.ConnectTimeout("slow"))
    with pytest.raises(httpx.ConnectTimeout):
        get_with_retry(URL, attempts=2)
    assert route.call_count == 2


@respx.mock
def test_backoff_doubles(no_retry_sleep):
    respx.get(URL).mock(return_value=httpx.Response(500))
    get_with_retry(URL, attempts=3)
    first, second = no_retry_sleep
    assert 1.0 <= first < 1.5 and 2.0 <= second < 2.5   # 1s, 2s (+ up to 0.5s jitter)
