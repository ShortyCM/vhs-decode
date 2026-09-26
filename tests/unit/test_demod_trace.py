import json
import os

from vhsdecode.demod_trace import write_demod_trace


def test_demod_trace_writes_process_and_job_identity(tmp_path):
    trace = tmp_path / "demod.jsonl"

    write_demod_trace(trace, "worker_received", block=17, request=3)

    record = json.loads(trace.read_text(encoding="ascii"))
    assert record["event"] == "worker_received"
    assert record["pid"] == os.getpid()
    assert record["block"] == 17
    assert record["request"] == 3
    assert isinstance(record["time_ns"], int)
