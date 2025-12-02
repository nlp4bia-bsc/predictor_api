import pandas as pd
import numpy as np
from typing import cast, List
from datetime import date, datetime
import torch

def date_linear_impute(dates: list[datetime | None]) -> list[date]:
    s = pd.to_datetime(pd.Series(dates), errors="coerce")
    n = len(s)

    # All null -> [1,2,...,n]
    if s.dropna().count() < 1: # if only nulls or only one non-null (don't have reference to interpolate)
        return [date(2024 + (i // 12), (i % 12) + 1 , 1) for i in range(n)] # by default, do it monthly from Jan 1, 2024

    # No nulls -> return same values (as floats)
    if all([d is not None for d in dates]):
        return cast(List[date], list(dates))

    s_int = s.apply(lambda x: x.value if pd.notna(x) else np.nan).astype("float64")  # convert Timestamp -> integer ns since epoch (use float to allow NaN)
    # Linear interpolation, allow extrapolation at ends
    s_interp = s_int.interpolate(method="linear", limit_direction="both").tolist()
    
    # there is at least 2 non-nulls, so we can extrapolate
    if not dates[0]:
        first_valid_index = s.first_valid_index()
        assert type (first_valid_index) is int
        step = s_interp[first_valid_index + 1] - s_interp[first_valid_index]
        for i in range(first_valid_index - 1, -1, -1):
            s_interp[i] = s_interp[i + 1] - step

    if not dates[-1]:
        last_valid_index = s.last_valid_index()
        assert type (last_valid_index) is  int
        step = s_interp[last_valid_index] - s_interp[last_valid_index - 1]
        for i in range(last_valid_index + 1, n):
            s_interp[i] = s_interp[i - 1] + step
    
    dt_series = pd.to_datetime(s_interp)
    return [pd.Timestamp(x).date() for x in dt_series.tolist()]

def dates_to_log_deltas(case_dates: list[date]) -> list[tuple[float, float]]:
    """
    Convert one case's ordered dates into two differnce arrays:
      - log_prev: log1p(delta since previous visit)  (first visit -> 0)
      - log_start: log1p(delta since first visit)    (first visit -> 0)

    Returns:
      list of tuples [(log_prev0, log_start0), ...] length == len(case_dates)
    """
    first = case_dates[0]
    prev = case_dates[0]

    out = []
    for dt in case_dates:
        # delta from previous (in days, possibly fractional)
        delta_prev_seconds = (dt - prev).total_seconds()
        delta_prev = delta_prev_seconds / 86400.0 # assuming days are the unit

        # delta from first
        delta_start_seconds = (dt - first).total_seconds()
        delta_start = delta_start_seconds / 86400.0

        # first visit: if dt == first then delta_prev may be 0.0, keep that
        log_prev = float(torch.log1p(torch.tensor(delta_prev, dtype=torch.float32)).item())
        log_start = float(torch.log1p(torch.tensor(delta_start, dtype=torch.float32)).item())

        out.append((log_prev, log_start))
        prev = dt

    return out