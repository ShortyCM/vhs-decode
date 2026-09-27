import atexit
import ctypes
import functools
import multiprocessing
import os
import statistics
import sys
import threading
import time

import numpy as np
from llvmlite import binding, ir
from numba import njit, types
from numba.core import cgutils
from numba.extending import intrinsic

LABELS = (
    'Field wow map (inclusive)',
    'Chroma TBC (inclusive)',
    'Scale level preparation (inclusive)',
    'Discarded field metrics (inclusive)',
    'White IRE mean / reuse',
    'AGC white difference',
    'Debug inverse FFT / reuse',
    'Philips constants (overlaps 17)',
    'Fixed wow coordinate grids',
    'Chroma burst area',
    'Fixed FSC values / reuse',
    'Wow outlier preparation (inside 03)',
    'Pulse geometry / reuse',
    'Pulse filter kernel / reuse',
    'Timing dictionary initialization',
    'HSync/EQ/VSync timing calculations',
    'Philips time conversions (overlaps 08)',
    'Chroma pixel coordinates / reuse',
)
PROFILE_VARIANT = 'latest'
PROFILE_PARENT = 'b40fd8505a09788fd5ae7dfc27f71d8ac5c975f7'
_local = threading.local()
_registry = []
_registry_lock = threading.Lock()
_started = time.perf_counter_ns()

if os.name == 'nt':
    _clock_library = ctypes.WinDLL('kernel32', use_last_error=True)
    _clock_function = _clock_library.QueryPerformanceCounter
    _clock_function.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    _clock_function.restype = ctypes.c_int
    _frequency = ctypes.c_int64()
    _frequency_function = _clock_library.QueryPerformanceFrequency
    _frequency_function.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    _frequency_function.restype = ctypes.c_int
    if not _frequency_function(ctypes.byref(_frequency)):
        raise OSError('QueryPerformanceFrequency failed')
    NATIVE_NS_PER_TICK = 1e9 / _frequency.value
    binding.add_symbol('vhs_profile_qpc', ctypes.cast(_clock_function, ctypes.c_void_p).value)
else:
    _clock_library = ctypes.CDLL(None)
    _clock_function = _clock_library.clock_gettime
    NATIVE_NS_PER_TICK = 1.0
    _monotonic_clock = time.CLOCK_MONOTONIC
    binding.add_symbol('vhs_profile_clock_gettime', ctypes.cast(_clock_function, ctypes.c_void_p).value)

@intrinsic
def native_ticks(typing_context):
    def codegen(context, builder, signature, args):
        i64 = ir.IntType(64)
        i32 = ir.IntType(32)
        if os.name == 'nt':
            slot = cgutils.alloca_once(builder, i64)
            fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(i32, [i64.as_pointer()]), 'vhs_profile_qpc')
            builder.call(fn, [slot])
            return builder.load(slot)
        pair = ir.LiteralStructType([i64, i64])
        slot = cgutils.alloca_once(builder, pair)
        fn = cgutils.get_or_insert_function(builder.module, ir.FunctionType(i32, [i32, pair.as_pointer()]), 'vhs_profile_clock_gettime')
        builder.call(fn, [ir.Constant(i32, _monotonic_clock), slot])
        value = builder.load(slot)
        seconds = builder.extract_value(value, 0)
        nanoseconds = builder.extract_value(value, 1)
        return builder.add(builder.mul(seconds, ir.Constant(i64, 1000000000)), nanoseconds)
    return types.int64(), codegen

@njit(inline='always')
def native_record(buffer, stage, elapsed):
    row = stage - 1
    if buffer[row, 0] == 0:
        buffer[row, 2] = elapsed
        buffer[row, 4] = elapsed
    elif elapsed < buffer[row, 2]:
        buffer[row, 2] = elapsed
    buffer[row, 0] += 1
    buffer[row, 1] += elapsed
    if elapsed > buffer[row, 3]:
        buffer[row, 3] = elapsed

@njit
def _native_empty_samples(count):
    result = np.empty(count, dtype=np.int64)
    for i in range(count):
        start = native_ticks()
        result[i] = native_ticks() - start
    return result

def _buffers():
    found = getattr(_local, 'buffers', None)
    if found is None:
        found = (np.zeros((18, 5), dtype=np.int64), np.zeros((18, 5), dtype=np.int64), np.zeros((18, 5), dtype=np.int64))
        _local.buffers = found
        with _registry_lock:
            _registry.append(found)
    return found

def native_buffer():
    return _buffers()[1]

def _record(buffer, stages, elapsed):
    for stage in stages:
        row = stage - 1
        if buffer[row, 0] == 0:
            buffer[row, 2] = elapsed
            buffer[row, 4] = elapsed
        elif elapsed < buffer[row, 2]:
            buffer[row, 2] = elapsed
        buffer[row, 0] += 1
        buffer[row, 1] += elapsed
        buffer[row, 3] = max(buffer[row, 3], elapsed)

class region:
    __slots__ = ('stages', 'buffer', 'start')

    def __init__(self, stages, setup=False):
        self.stages = (stages,) if isinstance(stages, int) else stages
        self.buffer = _buffers()[2 if setup else 0]

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed = time.perf_counter_ns() - self.start
        _record(self.buffer, self.stages, elapsed)
        return False

def value(stages, operation):
    stages = (stages,) if isinstance(stages, int) else stages
    buffer = _buffers()[0]
    start = time.perf_counter_ns()
    try:
        return operation()
    finally:
        elapsed = time.perf_counter_ns() - start
        _record(buffer, stages, elapsed)

def measured(stages):
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            with region(stages):
                return function(*args, **kwargs)
        return wrapped
    return decorate

def reset():
    with _registry_lock:
        for buffers in _registry:
            for buffer in buffers:
                buffer.fill(0)

def snapshot():
    combined = np.zeros((18, 7), dtype=np.float64)
    with _registry_lock:
        for buffers in _registry:
            for kind, factor in ((0, 1.0), (1, NATIVE_NS_PER_TICK)):
                buffer = buffers[kind]
                for row in range(18):
                    count = int(buffer[row, 0])
                    if count:
                        combined[row, 0] += count
                        combined[row, 1] += buffer[row, 1] * factor
                        combined[row, 2] += buffer[row, 4] * factor
                        combined[row, 3] += 1
                        combined[row, 4] = max(combined[row, 4], buffer[row, 3] * factor)
            combined[:, 5] += buffers[2][:, 0]
            combined[:, 6] += buffers[2][:, 1]
    return combined

def report():
    if multiprocessing.current_process().name != 'MainProcess':
        return
    rows = snapshot()
    if not np.any(rows[:, 0]) and not np.any(rows[:, 5]):
        return
    empty = []
    expression = []
    for _ in range(2000):
        start = time.perf_counter_ns()
        empty.append(time.perf_counter_ns() - start)
        operation = lambda: None
        start = time.perf_counter_ns()
        operation()
        expression.append(time.perf_counter_ns() - start)
    native_floor = float(np.median(_native_empty_samples(2000))) * NATIVE_NS_PER_TICK
    print('\nRedundant-operations profile: ' + PROFILE_VARIANT, flush=True)
    print('Parent: ' + PROFILE_PARENT)
    print('Runtime measurements include first calls; first-call sums are per thread and timer kind.')
    print('Rest = total minus first-call sums. Native timers exclude JIT compilation; Python scopes may include it.')
    print('Setup counts and milliseconds are separate. Unreached or removed code reports zero calls.')
    print('Overlapping scopes: 02 includes chroma work in 01/03; 01 includes 09; 03 includes 12; 04 includes 05; 08 overlaps 17.')
    print('01 measures wow-map construction, whose original cache was superseded by explicit chroma reuse.')
    print('Do not add rows together or treat summed thread time as elapsed decode time.')
    print('Raw timer floors, ns: Python clock pair %.1f; clock pair plus empty expression %.1f; native clock pair %.1f.' % (statistics.median(empty), statistics.median(expression), native_floor))
    print('Instrumentation overhead is not subtracted. Tiny intervals need cautious interpretation; decode FPS is instrumented.')
    print('%2s  %-43s %10s %12s %12s %12s %12s %10s %12s' % ('ID', 'Stage', 'Calls', 'Total ms', 'First ms', 'Rest ms', 'Rest us/call', 'Setup n', 'Setup ms'))
    for index, row in enumerate(rows):
        count, total, first, first_count, maximum, setup_count, setup_time = row
        remaining = max(0.0, total - first)
        remaining_count = count - first_count
        mean = remaining / remaining_count / 1000 if remaining_count else 0.0
        print('%02d  %-43s %10d %12.3f %12.3f %12.3f %12.3f %10d %12.3f' % (index + 1, LABELS[index], count, total / 1e6, first / 1e6, remaining / 1e6, mean, setup_count, setup_time / 1e6))
    sys.stdout.flush()

atexit.register(report)
