#!/usr/bin/env python3
"""Validate process distribution and overlap in a production DEMOD trace."""

import json
import sys


def main(path, minimum_workers):
    with open(path, encoding="ascii") as trace_file:
        records = sorted(
            (json.loads(line) for line in trace_file), key=lambda record: record["time_ns"]
        )

    starts = {}
    intervals = []
    for record in records:
        key = (record.get("pid"), record.get("block"), record.get("request"))
        if record["event"] == "demodblock_started":
            starts[key] = record["time_ns"]
        elif record["event"] == "demodblock_finished" and key in starts:
            intervals.append((key[0], starts.pop(key), record["time_ns"]))

    worker_pids = {pid for pid, _, _ in intervals}
    assert len(worker_pids) >= minimum_workers, (
        f"real DEMOD work used {len(worker_pids)} worker PID(s), expected at least "
        f"{minimum_workers}: {sorted(worker_pids)}"
    )
    assert any(
        pid_a != pid_b and start_a < end_b and start_b < end_a
        for pid_a, start_a, end_a in intervals
        for pid_b, start_b, end_b in intervals
    ), "real DEMOD work did not overlap between worker processes"

    print(
        f"real DEMOD work overlapped across {len(worker_pids)} worker PIDs: "
        f"{sorted(worker_pids)}"
    )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
