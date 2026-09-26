import multiprocessing
import threading

from lddecode.core import DemodCache


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


def test_demodcache_uses_process_workers():
    cache = _IdleDemodCache(_FakeRF(), None, None, None, num_worker_threads=2)
    try:
        assert len(cache.threads) == 2
        assert all(isinstance(worker, multiprocessing.process.BaseProcess) for worker in cache.threads)
        assert all(not isinstance(worker, threading.Thread) for worker in cache.threads)
        assert all(worker.is_alive() for worker in cache.threads)
    finally:
        cache.end()


def test_demodcache_zero_workers_remains_synchronous():
    cache = _IdleDemodCache(_FakeRF(), None, None, None, num_worker_threads=0)
    try:
        assert cache.threads == []
        assert cache._process_context is None
    finally:
        cache.end()
