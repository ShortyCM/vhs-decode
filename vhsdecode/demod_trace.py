"""Low-overhead, opt-in tracing for the multiprocessing demodulation path."""

import json
import os
import time


def write_demod_trace(path, event, *, block=None, request=None):
    """Append one atomic JSON-lines event to *path*.

    ``monotonic_ns`` is shared by all processes on the supported platforms, so
    events can be sorted without relying on wall-clock synchronization.  A
    separate open/write/close keeps the helper safe after ``fork`` and avoids
    passing an unpickleable file object to spawn workers.
    """
    if not path:
        return

    record = {
        "time_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "event": event,
    }
    if block is not None:
        record["block"] = block
    if request is not None:
        record["request"] = request

    data = (json.dumps(record, separators=(",", ":")) + "\n").encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
