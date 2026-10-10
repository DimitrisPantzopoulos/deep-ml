import numpy as np

class ZeroCopyBatchLoader:
    def __init__(self, data: np.ndarray, batch_size: int):
        """Store data in a flat contiguous buffer simulating shared memory."""
        self.buffer : np.ndarray = np.ascontiguousarray(data).reshape(-1)

        self.n_samples, self.n_features = data.shape

        self.n_batches  : int = int(np.ceil(self.n_samples / batch_size))
        self.batch_size : int = batch_size

    def num_batches(self) -> int:
        """Return total number of batches."""
        return self.n_batches

    def get_batch(self, batch_idx: int) -> np.ndarray:
        """Return batch as a zero-copy view into the buffer."""
        if batch_idx <= 0 and batch_idx > self.num_batches():
            raise ValueError("Invalid batch_idx.")

        row_start : int = batch_idx * self.batch_size
        row_end   : int = min(row_start + self.batch_size, self.n_samples)

        batch_start  : int = row_start * self.n_features
        batch_end    : int = row_end * self.n_features
        batch_rows   : int = row_end - row_start

        return self.buffer[batch_start : batch_end].reshape(batch_rows, self.n_features)

    def is_zero_copy(self, batch_idx: int) -> bool:
        """Check whether the batch shares memory with the buffer."""
        return np.shares_memory(self.get_batch(batch_idx), self.buffer)

    def get_batch_means(self) -> list:
        """Return list of per-batch mean values, each rounded to 4 decimals."""
        
        return [round(np.mean(self.get_batch(batch_idx=i)), 4) for i in range(self.n_batches)]

    def write_to_buffer(self, row: int, col: int, value: float) -> None:
        """Write a value directly into the flat buffer at (row, col)."""
        buffer_idx : int = row * self.n_features + col

        self.buffer[buffer_idx] = value

