"""
Minimal test case demonstrating line_profiler empty file bug.

PROBLEM:
When using line_profiler with multiprocessing (DataLoader workers) and LINE_PROFILE=1,
multiple processes try to write to the same output files simultaneously, possibly causing race conditions.
This results in some profile files being created with only headers and no actual profile data.

HOW TO REPRODUCE:
1. Set environment variable: $env:LINE_PROFILE=1
2. Run: python line_profiler_test.py
3. Check directory for profile_output* files

EXPECTED RESULTS:
- Multiple profile files are created with timestamps
- Some files contain only "Timer unit: 1e-07 s" (empty files)
- Some files contain full profiling data
- Typically 5+ empty files per run due to race conditions

TECHNICAL DETAILS:
- Single @profile decorated function with multiprocessing DataLoader
- 5 iterations within one function call
- num_workers=8 creates multiple processes competing for file writes
- Race conditions (possibly) cause some processes to write incomplete/empty files
"""

import torch
from line_profiler import profile
from torch.utils.data import DataLoader, Dataset


class D(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, i):
        return torch.randn(2, 2), torch.randn(2, 2)


@profile
def test():
    dataset = D()
    loader = DataLoader(dataset, batch_size=1, num_workers=8)

    # Create multiple iterations within one function (one empty file per iteration)
    for i in range(5):
        print(f"Iter {i}")

        # Access dataloader (error does not trigger if removed)
        for _, _ in loader:
            pass


if __name__ == "__main__":
    test()
