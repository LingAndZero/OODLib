from tqdm import tqdm


class TqdmFileWrapper:
    def __init__(self, f, total_bytes, desc="Loading"):
        self.f = f
        self.pbar = tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc,
        )

    def read(self, size=-1):
        data = self.f.read(size)
        if data:
            self.pbar.update(len(data))
        return data

    def readline(self, size=-1):
        data = self.f.readline(size)
        if data:
            self.pbar.update(len(data))
        return data

    def close(self):
        self.pbar.close()
        return self.f.close()

    def __getattr__(self, name):
        return getattr(self.f, name)
