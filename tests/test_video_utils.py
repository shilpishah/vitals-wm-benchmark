"""vitals/adapters/video_utils.py::with_one_retry/call_with_timeout --
pure function, no GPU/network needed (2026-08, extracted from run_model_
population.py after a real incident: an unhandled remote timeout on ONE
episode crashed an entire n=5 population, losing 4 already-successful
episodes' real GPU cost along with it). call_with_timeout added after a
SECOND real incident on the same n=80 re-run: a call that hangs without
ever raising blocks with_one_retry's own try/except forever -- these
tests use time.sleep as the hanging stand-in, so they must run fast
(short timeouts) to stay a real unit test, not an integration test."""
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vitals.adapters.video_utils import with_one_retry, call_with_timeout, HungCallTimeout


def test_succeeds_first_try_without_retrying():
    calls = []

    def fn():
        calls.append(1)
        return "ok"

    result, exc = with_one_retry(fn)
    assert result == "ok" and exc is None
    assert len(calls) == 1, "must not call fn a second time when the first attempt succeeds"


def test_succeeds_on_retry_after_one_failure():
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("transient")
        return "ok"

    result, exc = with_one_retry(fn)
    assert result == "ok" and exc is None
    assert len(calls) == 2


def test_gives_up_after_two_consecutive_failures_without_raising():
    calls = []

    def fn():
        calls.append(1)
        raise RuntimeError(f"fail #{len(calls)}")

    result, exc = with_one_retry(fn)
    assert result is None
    assert isinstance(exc, RuntimeError)
    assert len(calls) == 2, "must retry EXACTLY once, not loop forever"


def test_call_with_timeout_returns_normally_when_fast_enough():
    result = call_with_timeout(lambda: "ok", timeout_s=2.0)
    assert result == "ok"


def test_call_with_timeout_raises_hung_call_timeout_for_a_call_that_never_returns():
    def hangs():
        time.sleep(5.0)
        return "too late"

    t0 = time.time()
    try:
        call_with_timeout(hangs, timeout_s=0.3)
        assert False, "must raise HungCallTimeout, not block until the hang finishes"
    except HungCallTimeout:
        pass
    elapsed = time.time() - t0
    assert elapsed < 2.0, f"must return control near the timeout deadline (0.3s), not wait out the full hang -- took {elapsed:.2f}s"


def test_with_one_retry_recovers_from_a_hang_on_first_attempt_only():
    """A call that hangs on attempt 1 but returns normally on attempt 2 --
    the exact shape of the real incident this was built for (a stuck
    container on one try, a fresh one on retry)."""
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 1:
            time.sleep(5.0)
        return "ok"

    result, exc = with_one_retry(fn, timeout_s=0.3)
    assert result == "ok" and exc is None
    assert len(calls) == 2


def test_with_one_retry_gives_up_on_two_consecutive_hangs_without_blocking_forever():
    def always_hangs():
        time.sleep(5.0)
        return "never"

    t0 = time.time()
    result, exc = with_one_retry(always_hangs, timeout_s=0.3)
    elapsed = time.time() - t0
    assert result is None
    assert isinstance(exc, HungCallTimeout)
    assert elapsed < 3.0, (f"must give up after ~2x the per-attempt timeout (0.6s), not silently wait "
                           f"out both 5s hangs -- took {elapsed:.2f}s")


if __name__ == "__main__":
    test_succeeds_first_try_without_retrying()
    print("PASS  test_succeeds_first_try_without_retrying")
    test_succeeds_on_retry_after_one_failure()
    print("PASS  test_succeeds_on_retry_after_one_failure")
    test_gives_up_after_two_consecutive_failures_without_raising()
    print("PASS  test_gives_up_after_two_consecutive_failures_without_raising")
    test_call_with_timeout_returns_normally_when_fast_enough()
    print("PASS  test_call_with_timeout_returns_normally_when_fast_enough")
    test_call_with_timeout_raises_hung_call_timeout_for_a_call_that_never_returns()
    print("PASS  test_call_with_timeout_raises_hung_call_timeout_for_a_call_that_never_returns")
    test_with_one_retry_recovers_from_a_hang_on_first_attempt_only()
    print("PASS  test_with_one_retry_recovers_from_a_hang_on_first_attempt_only")
    test_with_one_retry_gives_up_on_two_consecutive_hangs_without_blocking_forever()
    print("PASS  test_with_one_retry_gives_up_on_two_consecutive_hangs_without_blocking_forever")
