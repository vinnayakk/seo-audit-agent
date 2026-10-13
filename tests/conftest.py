import pytest

from seo_audit_agent.fetcher import get_page


@pytest.fixture(autouse=True)
def clear_page_cache():
    get_page.cache_clear()
    yield
    get_page.cache_clear()