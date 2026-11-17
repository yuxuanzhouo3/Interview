"""
LC 2650. Design Cancellable Function — Hard
https://leetcode.com/problems/design-cancellable-function/
Best: Generator driver with cancel race — O(steps)
"""

from typing import Any, Callable, Generator, Tuple
import threading

class Cancelled(Exception):
    pass

def cancellable(
    generator: Generator[Any, Any, Any],
) -> Tuple[Callable[[], None], "FutureResult"]:
    """
    Python analog of LC 2650 (JS async generator cancel).
    Yields are treated as 'awaitables' that may be interrupted by cancel().
    """
    cancelled = {"v": False}
    result = {"value": None, "error": None, "done": False}
    done_evt = threading.Event()

    def cancel():
        cancelled["v"] = True

    def run():
        try:
            try:
                nxt = next(generator)
            except StopIteration as e:
                result["value"] = e.value
                result["done"] = True
                done_evt.set()
                return
            while True:
                if cancelled["v"]:
                    try:
                        nxt = generator.throw(Cancelled("Cancelled"))
                    except StopIteration as e:
                        result["value"] = e.value
                        result["done"] = True
                        done_evt.set()
                        return
                    except Exception as e:
                        result["error"] = e
                        result["done"] = True
                        done_evt.set()
                        return
                    continue
                # "await" yielded value: if callable, call it; else use as-is
                try:
                    if callable(nxt):
                        val = nxt()
                    else:
                        val = nxt
                    nxt = generator.send(val)
                except StopIteration as e:
                    result["value"] = e.value
                    result["done"] = True
                    done_evt.set()
                    return
                except Exception as e:
                    try:
                        nxt = generator.throw(e)
                    except StopIteration as se:
                        result["value"] = se.value
                        result["done"] = True
                        done_evt.set()
                        return
                    except Exception as e2:
                        result["error"] = e2
                        result["done"] = True
                        done_evt.set()
                        return
        except Exception as e:
            result["error"] = e
            result["done"] = True
            done_evt.set()

    class FutureResult:
        def result(self, timeout=None):
            done_evt.wait(timeout)
            if result["error"] is not None:
                raise result["error"]
            return result["value"]

        @property
        def done(self):
            return result["done"]

    t = threading.Thread(target=run, daemon=True)
    t.start()
    # For deterministic tests without real async, also expose sync runner
    return cancel, FutureResult()

def run_sync(gen_factory, cancel_after_steps=None):
    """Deterministic helper for tests: drive generator in-thread with optional cancel."""
    g = gen_factory()
    cancelled = False
    steps = 0
    try:
        nxt = next(g)
        while True:
            steps += 1
            if cancel_after_steps is not None and steps >= cancel_after_steps:
                cancelled = True
                try:
                    nxt = g.throw(Cancelled("Cancelled"))
                except StopIteration as e:
                    return e.value
                except Cancelled as e:
                    raise
                continue
            val = nxt() if callable(nxt) else nxt
            try:
                nxt = g.send(val)
            except StopIteration as e:
                return e.value
    except Cancelled:
        raise

class Solution:
    def cancellable(self, generator):
        return cancellable(generator)

def _test():

    def gen_ok():
        x = yield (lambda: 40)
        y = yield (lambda: 2)
        return x + y

    assert run_sync(gen_ok) == 42

    def gen_err():
        yield (lambda: (_ for _ in ()).throw(ValueError("Hello")))
        return 1

    try:
        run_sync(gen_err)
        assert False
    except ValueError as e:
        assert str(e) == "Hello"

    def gen_cancel():
        yield (lambda: 1)
        yield (lambda: 2)
        return 3

    try:
        run_sync(gen_cancel, cancel_after_steps=1)
        assert False
    except Cancelled as e:
        assert str(e) == "Cancelled"
    print("LC 2650 OK")

if __name__ == "__main__":
    _test()
