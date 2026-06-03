'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
LRU Cache Utility Module.

Bounded least-recently-used cache used by the shared geometry/beam
memoization layers. Behaves as a drop-in replacement for a plain dict
(get / __setitem__ / __contains__ / len / clear) but evicts the
least-recently-used entry once the configured maxsize is exceeded, so a
long DE sweep cannot grow the caches without bound.

Both reads (get) and writes (__setitem__) mark the touched key as most
recently used. A maxsize <= 0 disables eviction (unbounded dict).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from collections import OrderedDict
from typing import Any

# ================ Module imports ================


# ================================================================================
# PUBLIC API - Bounded LRU cache
# ================================================================================

class LRUCache(OrderedDict):
    '''
    Least-recently-used cache with a fixed entry ceiling.

    Property    Size        Description                             Units
    --------    --------    ------------------------------------    --------
    maxsize     (1,)        Max entries kept; <= 0 means unbounded   -
    '''

    def __init__(self, maxsize: int = 0, *args, **kwargs) -> None:
        '''
        Args:
            maxsize: Maximum number of entries to retain. When the cache
                grows past this, the least-recently-used entry is dropped.
                A value <= 0 disables eviction (behaves as a plain dict).
        '''
        self.maxsize = int(maxsize)
        super().__init__(*args, **kwargs)

    def get(self, key: Any, default: Any = None) -> Any:
        '''Return value for key (marking it most-recent), or default on miss.'''
        if key in self:
            self.move_to_end(key)
            return super().__getitem__(key)
        return default

    def __setitem__(self, key: Any, value: Any) -> None:
        '''Insert/update key as most-recent and evict the oldest if over size.'''
        if key in self:
            super().__setitem__(key, value)
            self.move_to_end(key)
            return
        super().__setitem__(key, value)
        if self.maxsize > 0 and len(self) > self.maxsize:
            super().__delitem__(next(iter(self)))
