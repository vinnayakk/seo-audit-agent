import os

# Tests must never send traces to Langfuse, even when .env holds real keys.
# load_dotenv() does not override variables that are already set, so this wins.
os.environ["LANGFUSE_TRACING_ENABLED"] = "false"

import pytest

from seo_audit_agent.fetcher import get_page


@pytest.fixture(autouse=True)
def clear_page_cache():
    get_page.cache_clear()
    yield
    get_page.cache_clear()