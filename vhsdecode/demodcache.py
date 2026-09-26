import multiprocessing
import threading
import time
import types
from multiprocessing import shared_memory

import numpy as np

from lddecode.core import DemodCache
from vhsdecode.addons.gnuradioZMQ import ZMQSend, ZMQReceive


_WORKER_RF_ATTRIBUTES = (
    "DecoderParams",
    "Filters",
    "_chroma_trap",
    "_disable_diff_demod",
    "_do_cafc",
    "_high_boost",
    "_notch",
    "_options",
    "_sub_emphasis_params",
    "_use_fsc_notch_filter",
    "_video_eq",
    "blockcut",
    "blockcut_end",
    "blocklen",
    "chromaTrap",
    "freq_hz",
    "system",
    "PAL_V4300D_NotchFilter",
    "freq",
)


def _plain_value(value):
    if hasattr(value, "_asdict"):
        return types.SimpleNamespace(**value._asdict())
    return value


def _make_worker_rf_state(rf):
    state = {}
    for attribute in _WORKER_RF_ATTRIBUTES:
        if hasattr(rf, attribute):
            state[attribute] = _plain_value(getattr(rf, attribute))
    state["debug_plot"] = None
    return state


def _vhs_demod_process(worker_index, input_queue, output_queue, rf_class, rf_state):
    rf = rf_class.__new__(rf_class)
    rf.__dict__.update(rf_state)
    result_segments = {}

    while True:
        item = input_queue.get()

        if item is None or item[0] == "END":
            for result_shm in result_segments.values():
                try:
                    result_shm.close()
                finally:
                    try:
                        result_shm.unlink()
                    except FileNotFoundError:
                        pass
            return

        if item[0] == "RELEASE_RESULT":
            result_name = item[1]
            result_shm = result_segments.pop(result_name, None)
            if result_shm is not None:
                try:
                    result_shm.close()
                finally:
                    try:
                        result_shm.unlink()
                    except FileNotFoundError:
                        pass
            continue

        if item[0] == "DEMOD_SHM":
            _, token, blocknum, shm_name, shape, dtype_str, request = item
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                data = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
                demod = rf.demodblock(
                    data=data,
                    fftdata=None,
                    mtf_level=0,
                    cut=True,
                )
            finally:
                shm.close()

            video = np.ascontiguousarray(demod["video"])
            result_shm = shared_memory.SharedMemory(create=True, size=video.nbytes)
            result = np.ndarray(video.shape, dtype=video.dtype, buffer=result_shm.buf)
            result[...] = video
            result_segments[result_shm.name] = result_shm
            output_queue.put(
                (
                    "DEMOD_RESULT_SHM",
                    worker_index,
                    blocknum,
                    token,
                    result_shm.name,
                    video.shape,
                    video.dtype.descr,
                    request,
                )
            )

        elif item[0] == "SYNC_SHM":
            _, token, blocknum, shm_name, shape, dtype_str = item
            shm = shared_memory.SharedMemory(name=shm_name)
            try:
                data = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
                sync = np.ascontiguousarray(
                    rf.demodblock_sync(
                        data=data,
                        fftdata=None,
                        cut=True,
                    )
                )
            finally:
                shm.close()

            result_shm = shared_memory.SharedMemory(create=True, size=sync.nbytes)
            result = np.ndarray(sync.shape, dtype=sync.dtype, buffer=result_shm.buf)
            result[...] = sync
            result_segments[result_shm.name] = result_shm
            output_queue.put(
                (
                    "SYNC_RESULT_SHM",
                    worker_index,
                    blocknum,
                    token,
                    result_shm.name,
                    sync.shape,
                    sync.dtype.str,
                )
            )


class _VHSJobDispatcher:
    def __init__(self, worker_queues):
        self.worker_queues = worker_queues
        self.next_worker = 0
        self.next_token = 0
        self.shared_inputs = {}

    def _select_queue(self):
        queue = self.worker_queues[self.next_worker]
        self.next_worker = (self.next_worker + 1) % len(self.worker_queues)
        return queue

    def _share_array(self, array):
        array = np.ascontiguousarray(array)
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        target = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        target[...] = array

        token = self.next_token
        self.next_token += 1
        self.shared_inputs[token] = shm
        return token, shm.name, array.shape, array.dtype.str

    def put(self, item):
        if item is None:
            self._select_queue().put(None)
            return

        if isinstance(item, tuple) and len(item) == 4 and not isinstance(item[0], str):
            blocknum, block, mtf, request = item
            item = ("DEMOD", blocknum, block, mtf, request)

        kind = item[0]

        if kind == "DEMOD":
            blocknum, block, _, request = item[1:]
            token, name, shape, dtype_str = self._share_array(block["rawinput"])
            self._select_queue().put(
                ("DEMOD_SHM", token, blocknum, name, shape, dtype_str, request)
            )
            return

        if kind == "SYNC":
            blocknum, block = item[1:]
            token, name, shape, dtype_str = self._share_array(block["rawinput"])
            self._select_queue().put(
                ("SYNC_SHM", token, blocknum, name, shape, dtype_str)
            )
            return

        if kind == "NEWPARAMS":
            for queue in self.worker_queues:
                queue.put(item)
            return

        self._select_queue().put(item)

    def qsize(self):
        total = 0
        for queue in self.worker_queues:
            try:
                total += queue.qsize()
            except (NotImplementedError, AttributeError):
                pass
        return total

    def release(self, token):
        shm = self.shared_inputs.pop(token, None)
        if shm is None:
            return
        try:
            shm.close()
        finally:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def release_result(self, worker_index, shm_name):
        self.worker_queues[worker_index].put(("RELEASE_RESULT", shm_name))

    def close(self):
        for token in list(self.shared_inputs):
            self.release(token)


class _VHSResultQueue:
    def __init__(self, queue, dispatcher):
        self.queue = queue
        self.dispatcher = dispatcher

    def get(self):
        result = self.queue.get()
        if result is None:
            return None

        if isinstance(result, tuple) and result:
            if result[0] == "DEMOD_RESULT_SHM":
                (
                    _,
                    worker_index,
                    blocknum,
                    token,
                    shm_name,
                    shape,
                    dtype_descr,
                    request,
                ) = result
                self.dispatcher.release(token)

                shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    video = np.ndarray(
                        shape,
                        dtype=np.dtype(dtype_descr),
                        buffer=shm.buf,
                    ).copy().view(np.recarray)
                finally:
                    shm.close()
                    self.dispatcher.release_result(worker_index, shm_name)

                return (
                    blocknum,
                    {
                        "demod": {"video": video},
                        "request": request,
                        "MTF": 0,
                    },
                )

            if result[0] == "SYNC_RESULT_SHM":
                _, worker_index, blocknum, token, shm_name, shape, dtype_str = result
                self.dispatcher.release(token)

                shm = shared_memory.SharedMemory(name=shm_name)
                try:
                    sync = np.ndarray(
                        shape,
                        dtype=np.dtype(dtype_str),
                        buffer=shm.buf,
                    ).copy()
                finally:
                    shm.close()
                    self.dispatcher.release_result(worker_index, shm_name)

                return blocknum, {"sync": sync}

            if len(result) == 3:
                blocknum, output, token = result
                self.dispatcher.release(token)
                return blocknum, output

        return result

    def put(self, item):
        self.queue.put(item)


class DemodCacheTape(DemodCache):
    def __init__(self, *args, **kwargs):
        rf = args[0]
        self._gnrc_afe = rf.options.gnrc_afe

        base_args = list(args)
        if "num_worker_threads" in kwargs:
            requested_workers = kwargs["num_worker_threads"]
        elif len(base_args) >= 6:
            requested_workers = base_args[5]
        else:
            requested_workers = 6

        if requested_workers <= 0 or self._gnrc_afe:
            super(DemodCacheTape, self).__init__(*args, **kwargs)
            if self._gnrc_afe:
                self.zmqsend = ZMQSend()
                self.zmqreceive = ZMQReceive()
                print(
                    "Open GNURadio with ZMQ REQ source set at tcp://localhost:%d and ZMQ REP sink set at tcp://*:%d"
                    % (self.zmqsend.port, self.zmqreceive.port)
                )
                print(
                    "The data stream will be of the float type at 40MSPS (40MHz sample rate)"
                )
                print(
                    "It will send the raw RF for further processing prior to demodulation (useful for RF EQ discovery "
                    "and group delay compensation)"
                )
                print(
                    "You might want to do this in single threaded decode mode (-t 1 parameter) - TODO: might not work correctly with --no_resample yet."
                )
            return

        if "num_worker_threads" in kwargs:
            kwargs = dict(kwargs)
            kwargs["num_worker_threads"] = 0
        elif len(base_args) >= 6:
            base_args[5] = 0
        else:
            kwargs = dict(kwargs)
            kwargs["num_worker_threads"] = 0

        super(DemodCacheTape, self).__init__(*base_args, **kwargs)

        self.q_out.put(None)
        self.deqeue_thread.join()

        self._process_context = multiprocessing.get_context()
        self._worker_queues = [
            self._process_context.Queue() for _ in range(requested_workers)
        ]
        process_output_queue = self._process_context.Queue()

        self._job_dispatcher = _VHSJobDispatcher(self._worker_queues)
        self.q_in = self._job_dispatcher
        self.q_out = _VHSResultQueue(process_output_queue, self._job_dispatcher)

        self.num_worker_threads = requested_workers
        self.threads = []

        rf_state = _make_worker_rf_state(self.rf)
        for worker_index, worker_queue in enumerate(self._worker_queues):
            process = self._process_context.Process(
                target=_vhs_demod_process,
                args=(
                    worker_index,
                    worker_queue,
                    process_output_queue,
                    self.rf.__class__,
                    rf_state,
                ),
                daemon=True,
            )
            process.start()
            self.threads.append(process)

        self.deqeue_thread = threading.Thread(target=self.dequeue, daemon=True)
        self.deqeue_thread.start()

    def end(self):
        if not self.ended:
            super(DemodCacheTape, self).end()
            if hasattr(self, "_job_dispatcher"):
                self._job_dispatcher.close()

    def worker(self, return_on_empty=False):
        blocksrun = 0
        blockstime = 0

        rf = self.rf

        while True:
            if return_on_empty and self.q_in.qsize() == 0:
                return

            item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, _, request = item[1:]

                if self._gnrc_afe:
                    raw_input = block["rawinput"]
                    raw_size = raw_input.size
                    self.zmqsend.send(raw_input)
                    block["rawinput"] = self.zmqreceive.receive(raw_size)

                output = {}

                if "fft" not in block:
                    fftdata = None
                else:
                    fftdata = block["fft"]

                st = time.time()
                output["demod"] = rf.demodblock(
                    data=block["rawinput"],
                    fftdata=fftdata,
                    mtf_level=0,
                    cut=True,
                )
                blockstime += time.time() - st
                blocksrun += 1

                output["request"] = request
                output["MTF"] = 0

                self.q_out.put((blocknum, output))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])
