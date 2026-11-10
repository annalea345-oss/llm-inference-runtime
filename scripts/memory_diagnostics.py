"""Experiment-only observers; never imported by the normal runtime factories."""

import ctypes
import gc
import sys
import tracemalloc
import weakref


class NativeAllocator:
    """Darwin public malloc-zone counters and optional best-effort pressure relief."""

    class Statistics(ctypes.Structure):
        _fields_ = [('blocks_in_use', ctypes.c_uint), ('size_in_use', ctypes.c_size_t),
                    ('max_size_in_use', ctypes.c_size_t), ('size_allocated', ctypes.c_size_t)]

    def __init__(self):
        if sys.platform != 'darwin':
            raise RuntimeError('native allocator diagnostics require macOS')
        self.library = ctypes.CDLL('/usr/lib/libSystem.dylib')
        self.library.malloc_zone_statistics.argtypes = [ctypes.c_void_p, ctypes.POINTER(self.Statistics)]
        self.library.malloc_zone_statistics.restype = None
        self.library.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.library.malloc_zone_pressure_relief.restype = ctypes.c_size_t

    def statistics(self):
        stats = self.Statistics()
        self.library.malloc_zone_statistics(None, ctypes.byref(stats))
        return {name: int(getattr(stats, name)) for name, _ in stats._fields_}

    def relieve(self):
        # Official API: NULL examines all zones; goal=0 requests maximal relief.
        return int(self.library.malloc_zone_pressure_relief(None, 0))


class WeakAudit:
    """Count object lifetimes without retaining requests, sequences, or tensors."""

    def __init__(self):
        self.references = {'requests': [], 'released_sequences': [], 'released_kv_tensors': []}
        self.observed = {name: 0 for name in self.references}

    def observe(self, kind, value):
        self.references[kind].append(weakref.ref(value))
        self.observed[kind] += 1

    def snapshot(self):
        result = {}
        for kind, refs in self.references.items():
            # Compact dead weakrefs each snapshot so the observer itself does
            # not accumulate one record for every completed request forever.
            refs[:] = [ref for ref in refs if ref() is not None]
            result[kind] = {'observed': self.observed[kind], 'alive': len(refs)}
        return result


def require_idle(scheduler):
    if scheduler.active or scheduler.queued_requests:
        raise RuntimeError('diagnostic operation requires an idle scheduler')


class MemoryProbe:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.executor = scheduler.executor
        self.audit = WeakAudit()
        self.allocator = NativeAllocator()
        self.weight_refs = [weakref.ref(p) for p in self.executor.model.parameters()]
        self.model_ref = weakref.ref(self.executor.model)
        self.anchor = None
        self.gc_collected_by_probe = 0
        original_submit = scheduler._submit
        original_release = self.executor.release_request

        def submit(*args, **kwargs):
            request = original_submit(*args, **kwargs)
            self.audit.observe('requests', request)
            return request

        def release(request_id):
            state = self.executor._states.get(request_id)
            if state is not None:
                self.audit.observe('released_sequences', state)
                if state.cache is not None:
                    for keys, values in state.cache:
                        self.audit.observe('released_kv_tensors', keys)
                        self.audit.observe('released_kv_tensors', values)
            original_release(request_id)

        # Instrument only this private profiling app's instances.
        scheduler._submit = submit
        self.executor.release_request = release

    def snapshot(self):
        require_idle(self.scheduler)
        traced = tracemalloc.get_traced_memory() if tracemalloc.is_tracing() else (None, None)
        return {'requests_registered': len(self.scheduler._requests),
                'active_sequences': self.executor.active_sequences,
                'reserved_tokens': self.scheduler.kv_cache.reserved_tokens,
                'used_tokens': self.scheduler.kv_cache.used_tokens,
                'kv_tensor_bytes': self.executor.cache_bytes,
                'weak_objects': self.audit.snapshot(),
                'model_alive': self.model_ref() is not None,
                'model_parameters_alive': sum(ref() is not None for ref in self.weight_refs),
                'python_traced_current_bytes': traced[0], 'python_traced_peak_bytes': traced[1],
                'tracemalloc_metadata_bytes': tracemalloc.get_tracemalloc_memory() if tracemalloc.is_tracing() else None,
                'native_malloc': self.allocator.statistics(), 'gc_counts': gc.get_count(),
                'gc_stats': gc.get_stats(), 'gc_collected_by_probe': self.gc_collected_by_probe}

    def operate(self, operation):
        require_idle(self.scheduler)
        if operation == 'start_trace':
            if tracemalloc.is_tracing():
                raise ValueError('tracing already enabled')
            tracemalloc.start(5)
        elif operation == 'anchor':
            self.anchor = tracemalloc.take_snapshot()
        elif operation == 'gc':
            self.gc_collected_by_probe += gc.collect()
        elif operation == 'relieve':
            released = self.allocator.relieve()
            return {'allocator_reported_released_bytes': released, 'snapshot': self.snapshot()}
        elif operation == 'diff':
            if self.anchor is None:
                raise ValueError('trace anchor is missing')
            difference = tracemalloc.take_snapshot().compare_to(self.anchor, 'lineno')
            return {'python_allocation_diff': [
                {'site': str(stat.traceback[-1]), 'bytes_delta': stat.size_diff,
                 'blocks_delta': stat.count_diff} for stat in difference[:20]],
                'snapshot': self.snapshot()}
        else:
            raise ValueError('unknown diagnostic operation')
        return {'snapshot': self.snapshot()}
