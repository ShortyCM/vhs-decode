Advanced flags
----

```--debug``` sets logger verbosity level to *debug*. Useful for debugging.

`--noAGC` disables the **A**utomatic **G**ain **C**ontrol feature, mainly affecting image brightness/gamma levels. Use if experiencing fluctuating brightness levels or overly dark/bright output.

`-ct` enables a *chroma trap*, a filter intended to reduce chroma interference on the main luma signal. Use if seeing banding or checkerboarding on the main luma .tbc in ld-analyse.

`-sl` defines the output *sharpness level*, as an integer from 0-100, default being 0. Higher values are better suited for plain, flat images i.e. cartoons and animated material, as strong ghosting can occur (akin to cranking up the sharpness on any regular TV set.)

`--notch, --notch_q` define the center frequency and Q factor for an (optional) built-in notch (bandpass) filter. Intended primarily for reducing noise from interference, though the main decoder logic already compensates for this accodring to each tape and TV system's specific frequency values.

`--doDOD` enables *dropout correction*. Please note, this does not force vhs-decode to perform dropout correction; instead, it adds a flag to the output .json, leaving it to be performed in the next step (running any of the gen_vid_chroma scripts.)

# Demodulation worker tracing

`--demod_trace PATH` records the actual VHS multiprocessing path during a
decode. The output is JSON Lines and is truncated when the decoder starts.
Each record contains a monotonic nanosecond timestamp, PID, block number,
request generation, and one of these events:

* `parent_queued` — the coordinator submitted the DEMOD job;
* `worker_received` — a worker process received it;
* `demodblock_started` and `demodblock_finished` — the entry and exit of
  `VHSRFDecode.demodblock` itself;
* `worker_returned` — the worker put the result on the output queue; and
* `parent_received` — the coordinator dequeue thread received the result.

For example:

```sh
vhs-decode --threads 8 --demod_trace demod.jsonl input.raw output
```

Sort records by `time_ns` before comparing them. The interval from
`parent_queued` to `worker_received` measures input-queue transport, the
`demodblock_started` to `demodblock_finished` interval is useful computation,
and `worker_returned` to `parent_received` measures result-queue transport.
Counts of `demodblock_started` grouped by PID show which workers actually do
useful work, while overlapping start/finish intervals show whether that work
runs concurrently.
