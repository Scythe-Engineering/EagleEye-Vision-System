from typing import Any


def _format_ms(seconds: float) -> float:
    """Convert seconds to milliseconds rounded to two decimals.

    Args:
        seconds: Duration in seconds.

    Returns:
        Duration in milliseconds as a float rounded to two decimals.
    """
    return round(seconds * 1000.0, 2)

def print_timing_summary(operations: list[Any], operation_time_history: list[list[float]], total_time_history: list[float], logger=None) -> None:
    """Print a readable CLI timing summary using a moving average of the last 50 runs.

    Args:
        operations (list[Any]): List of operations.
        operation_time_history (list[list[float]]): List of lists of operation times.
        total_time_history (list[float]): List of total pipeline times.
    """
    # Collect operation names for display in the table
    op_names = [type(op).__name__ for op in operations]
    # Compute moving-average (ms) for each operation from the last up-to-20 samples
    op_avgs_ms = [
        _format_ms(sum(hist) / len(hist)) if len(hist) > 0 else 0.0
        for hist in operation_time_history
    ]
    # Compute moving-average (ms) for total pipeline time
    total_avg_ms = (
        _format_ms(sum(total_time_history) / len(total_time_history))
        if len(total_time_history) > 0
        else 0.0
    )
    # Derive FPS from average total time
    fps_avg = (
        round(1000.0 / total_avg_ms, 2) if total_avg_ms > 0.0 else 0.0
    )
    # Determine column widths based on content for clean alignment
    name_col_width = max([len("Operation")] + [len(n) for n in op_names])
    time_col_width = max(len("Avg (ms)"), 8)
    # Build header and separators
    header = f"{'Operation'.ljust(name_col_width)} | {'Avg (ms)'.rjust(time_col_width)}"
    separator = f"{'-' * name_col_width}-+-{'-' * time_col_width}"
    # Build per-operation rows
    rows = [
        f"{name.ljust(name_col_width)} | {str(val).rjust(time_col_width)}"
        for name, val in zip(op_names, op_avgs_ms)
    ]
    # Build footer with totals and FPS
    footer_sep = f"{'=' * name_col_width}=+={'=' * time_col_width}"
    footer = f"{'Total'.ljust(name_col_width)} | {str(total_avg_ms).rjust(time_col_width)}"
    fps_line = f"{'FPS'.ljust(name_col_width)} | {str(fps_avg).rjust(time_col_width)}"
    # Compute overall average FPS across all recorded runs, not just the moving window
    overall_avg_time_ms = (
        _format_ms(sum(total_time_history) / len(total_time_history))
        if len(total_time_history) > 0
        else 0.0
    )
    overall_avg_fps = (
        round(1000.0 / overall_avg_time_ms, 2) if overall_avg_time_ms > 0.0 else 0.0
    )
    overall_fps_line = f"{'Total Avg FPS'.ljust(name_col_width)} | {str(overall_avg_fps).rjust(time_col_width)}"
    # Emit the table in a compact, readable format
    log_func = logger.log if logger else print
    log_func("\nTiming (moving average of last 50 runs)")
    log_func(header)
    log_func(separator)
    for row in rows:
        log_func(row)
    log_func(footer_sep)
    log_func(footer)
    log_func(fps_line)
    log_func(overall_fps_line)
    log_func("")
