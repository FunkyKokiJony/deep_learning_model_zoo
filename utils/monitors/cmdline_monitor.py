"""

"""
from tqdm import tqdm

class CmdLineMonitor:
    def __init__(self):
        self.callbacks = dict()
        self.tracking_stats = dict()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.clear()

    def add_callbacks(self, cb):
        self.callbacks[cb.get_name()] = cb

    def update(self, mode, idx, stats_dict=dict()):
        for _, callback in self.callbacks.items():
            callback(self, mode, idx, stats_dict)

    def add_stats(self, idx, stats_name, tag, val):
        if stats_name not in self.tracking_stats:
            self.tracking_stats[stats_name] = dict()

        self.tracking_stats[stats_name][tag] = val

    def display(self, title="=== Training Progress ==="):
        if (len(self.tracking_stats) == 0): return

        max_name_len = max(map(lambda k: len(k), self.tracking_stats.keys()))

        tqdm.write(title)

        for name, stats in self.tracking_stats.items():
            if (len(stats) == 0):
                continue

            _str = "{:<{width}} --- ".format(name, width=max_name_len)

            for tag, val in stats.items():
                _str += "{}: {}\t".format(tag, val)

            tqdm.write(_str)

        tqdm.write("")
        tqdm.write("")

    def reset(self):
        self.clear()

    def clear(self):
        self.tracking_stats.clear()

