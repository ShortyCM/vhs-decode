import multiprocessing
import os
import pickle
import threading
import time

import numpy as np

import pytest

from lddecode.core import DemodCache
from vhsdecode.compute_video_filters import (
    SubEmphasisParams,
    create_sub_emphasis_params,
)
from vhsdecode.demodcache import DemodCacheTape
from vhsdecode.process_types import Options, SysparamsConst


class _FakeRF:
    blocklen = 16
    blockcut = 2
    blockcut_end = 2
    freq_hz = 1_000
    SysParams = {"FPS": 25}


class _IdleDemodCache(DemodCache):
    """Minimal cache whose workers only exercise process lifecycle handling."""

    def worker(self, return_on_empty=False):
        while True:
            if self.q_in.get() is None:
                return


class _FakeVHSRF:
    pass


class _RecordingRF(_FakeRF):
    """RF stub that records which process executes each real DEMOD call."""

    def __init__(self, calls):
        self.calls = calls

    def demodblock(self, data, fftdata, mtf_level, cut):
        started = time.monotonic()
        # Long enough to make overlap unambiguous, while keeping the test fast.
        time.sleep(0.15)
        finished = time.monotonic()
        self.calls.append((os.getpid(), started, finished))
        return {"video": np.asarray(data[2:-2])}


def _paced_loader(_infile, offset, length):
    # Model a stateful streaming/resampling loader.  Publishing each job
    # immediately after this delay lets one fast consumer drain the queue.
    time.sleep(0.18)
    return np.arange(offset, offset + length, dtype=np.float64)


def test_tape_worker_rf_copy_only_contains_demodulation_state():
    rf = _FakeVHSRF()
    for attribute in DemodCacheTape._WORKER_RF_ATTRIBUTES:
        setattr(rf, attribute, attribute)
    rf.secam_servo_avg = threading.Lock()
    rf.decoder = threading.Lock()
    rf._processing_thread_pool = threading.Lock()
    rf.debug_plot = threading.Lock()

    cache = DemodCacheTape.__new__(DemodCacheTape)
    cache.rf = rf
    worker_rf = cache._make_worker_rf_copy()

    assert worker_rf.__dict__ == {
        attribute: attribute
        for attribute in DemodCacheTape._WORKER_RF_ATTRIBUTES
    } | {"debug_plot": None}
    assert pickle.loads(pickle.dumps(worker_rf)).__dict__ == worker_rf.__dict__


def test_vhs_decoder_namedtuples_are_pickleable_by_name():
    """Spawn workers must be able to import classes embedded in RF state."""
    options = Options(*range(len(Options._fields)))
    constants = SysparamsConst(*range(len(SysparamsConst._fields)))

    assert pickle.loads(pickle.dumps(options)) == options
    assert pickle.loads(pickle.dumps(constants)) == constants
    assert Options.__module__ == "vhsdecode.process_types"
    assert SysparamsConst.__module__ == "vhsdecode.process_types"


def test_sub_emphasis_params_are_pickleable_by_name():
    """The nonlinear filter configuration is included in spawned RF workers."""
    params = create_sub_emphasis_params({}, {}, 2, -40)

    assert pickle.loads(pickle.dumps(params)) == params
    assert type(params) is SubEmphasisParams
    assert SubEmphasisParams.__module__ == "vhsdecode.compute_video_filters"


def test_demodcache_uses_process_workers():
    cache = _IdleDemodCache(_FakeRF(), None, None, None, num_worker_threads=2)
    try:
        assert len(cache.threads) == 2
        assert all(isinstance(worker, multiprocessing.process.BaseProcess) for worker in cache.threads)
        assert all(not isinstance(worker, threading.Thread) for worker in cache.threads)
        assert all(worker.is_alive() for worker in cache.threads)
        assert len({worker.pid for worker in cache.threads}) == 2
        assert all(worker.pid != os.getpid() for worker in cache.threads)
    finally:
        cache.end()


def test_demodcache_zero_workers_remains_synchronous():
    cache = _IdleDemodCache(_FakeRF(), None, None, None, num_worker_threads=0)
    try:
        assert cache.threads == []
        assert cache._process_context is None
    finally:
        cache.end()


def test_demod_jobs_are_distributed_and_overlap_across_worker_processes():
    """Multiple PIDs must execute DEMOD, not merely wait beside one worker."""
    with multiprocessing.Manager() as manager:
        calls = manager.list()
        cache = DemodCache(
            _RecordingRF(calls),
            None,
            _paced_loader,
            None,
            num_worker_threads=4,
        )
        try:
            result = cache.read(0, cache.blocksize * 8)
            assert result is not None

            recorded = list(calls)
            worker_pids = {pid for pid, _, _ in recorded}
            assert len(worker_pids) >= 2
            assert worker_pids.issubset({worker.pid for worker in cache.threads})
            assert os.getpid() not in worker_pids

            # At least two DEMOD intervals from different processes overlap.
            assert any(
                pid_a != pid_b and start_a < end_b and start_b < end_a
                for pid_a, start_a, end_a in recorded
                for pid_b, start_b, end_b in recorded
            )
        finally:
            cache.end()


NON_FORK_START_METHODS = [
    method
    for method in ("spawn", "forkserver")
    if method in multiprocessing.get_all_start_methods()
]


@pytest.mark.parametrize("start_method", NON_FORK_START_METHODS)
def test_demodcache_uses_processes_without_fork(monkeypatch, start_method):
    process_context = multiprocessing.get_context(start_method)
    monkeypatch.setattr(multiprocessing, "get_context", lambda: process_context)

    rf = _FakeRF()
    rf.decoder = threading.Lock()
    rf._processing_thread_pool = threading.Lock()
    with open(__file__, "rb") as infile:
        cache = _IdleDemodCache(rf, infile, None, None, num_worker_threads=2)
        workers = cache.threads.copy()
        try:
            assert cache._process_context is process_context
            assert len(cache.threads) == 2
            assert all(
                isinstance(worker, multiprocessing.process.BaseProcess)
                for worker in cache.threads
            )
            assert all(worker.is_alive() for worker in cache.threads)
            assert len({worker.pid for worker in cache.threads}) == 2
            assert all(worker.pid != os.getpid() for worker in cache.threads)
        finally:
            cache.end()
        assert all(worker.exitcode == 0 for worker in workers)
