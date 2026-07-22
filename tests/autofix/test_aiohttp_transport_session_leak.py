import asyncio
import gc
import os
import sys
import warnings

import aiohttp
import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.llms.custom_httpx.aiohttp_transport import LiteLLMAiohttpTransport


def _create_session_on_separate_closed_loop() -> aiohttp.ClientSession:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _mk() -> aiohttp.ClientSession:
        return aiohttp.ClientSession()

    session = loop.run_until_complete(_mk())
    loop.close()
    return session


@pytest.mark.asyncio
async def test_stale_loop_session_is_closed_not_leaked():
    """
    Regression: "Unclosed client session".

    When _get_valid_client_session finds that the cached session belongs to a
    different (here: closed) event loop, it recreates the session for the
    current loop. The previous session must be released synchronously; the old
    behavior scheduled a fire-and-forget close on the wrong loop (or relied on
    GC), leaving the session marked open, which is exactly what triggers
    aiohttp's "Unclosed client session" warning and leaks its sockets.
    """
    stale_session = _create_session_on_separate_closed_loop()
    assert not stale_session.closed

    transport = LiteLLMAiohttpTransport(client=lambda: aiohttp.ClientSession())
    transport.client = stale_session

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_session = transport._get_valid_client_session()

        assert new_session is not stale_session, "expected a fresh session for the running loop"
        assert stale_session.closed, "stale session from the old loop leaked (was not closed)"

        del stale_session
        gc.collect()
        unclosed = [w for w in caught if "Unclosed client session" in str(w.message)]
        assert not unclosed, f"aiohttp emitted an unclosed-session warning: {[str(w.message) for w in unclosed]}"

    if not new_session.closed:
        await new_session.close()
