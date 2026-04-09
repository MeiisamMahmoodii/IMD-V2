import functools
from pathlib import Path
import pickle

def disk_cache(cache_dir: str = "data/metadata/cache"):
    base = Path(cache_dir)
    base.mkdir(parents=True, exist_ok=True)
    def deco(func):
        @functools.wraps(func)
        def wrap(*args, **kwargs):
            key = f"{func.__name__}_{hash((args, tuple(sorted(kwargs.items()))))}.pkl"
            path = base / key
            if path.exists():
                return pickle.loads(path.read_bytes())
            val = func(*args, **kwargs)
            path.write_bytes(pickle.dumps(val))
            return val
        return wrap
    return deco
