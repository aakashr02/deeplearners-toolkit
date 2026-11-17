"""
io.py
=====

CSV loading for NumCompute, built on ``numpy.genfromtxt``.

* ``read_csv`` loads a whole file into one 2D array.
* ``read_csv_chunks`` streams a file as a sequence of 2D arrays of at
  most ``chunk_size`` rows, so files larger than memory can be processed
  incrementally. Only one chunk of text is held in memory at a time.

Both functions share the same parsing options and missing-value policy:

* ``missing="fill"`` replaces empty or unparsable fields with
  ``missing_value`` (NaN by default).
* ``missing="skip"`` drops every row that contains a missing field.

Concatenating all chunks from ``read_csv_chunks`` gives exactly the array
returned by ``read_csv`` with the same options.
"""

import os

import numpy as np

from numcompute.utils import validate_int


_MISSING_POLICIES = ("fill", "skip")


def _validate_options(path, delimiter, missing, skip_header):
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be a string or path-like object")

    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such CSV file: {path!r}")

    if delimiter is not None and not isinstance(delimiter, str):
        raise TypeError("delimiter must be a string or None (whitespace)")

    if missing not in _MISSING_POLICIES:
        raise ValueError(
            f"missing must be 'fill' or 'skip'; received {missing!r}"
        )

    return validate_int(skip_header, "skip_header", minimum=0)


def _parse_lines(lines, delimiter, missing_value, dtype, missing):
    """Parse text lines into a 2D array and apply the missing policy."""
    # For "skip", parse with NaN so that missing fields can be detected.
    fill = np.nan if missing == "skip" else missing_value
    parse_dtype = float if missing == "skip" else dtype

    data = np.genfromtxt(
        lines,
        delimiter=delimiter,
        filling_values=fill,
        dtype=parse_dtype,
        ndmin=2,
    )

    if missing == "skip":
        data = data[~np.isnan(data).any(axis=1)]
        if dtype is not float:
            data = data.astype(dtype)

    return data


def read_csv(path, delimiter=",", missing_value=np.nan, dtype=float,
             missing="fill", skip_header=0):
    """
    Load a CSV file into a two-dimensional NumPy array.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV file.
    delimiter : str or None, default=","
        Field separator, e.g. "," or "\\t". None splits on whitespace.
    missing_value : scalar, default=np.nan
        Placeholder for missing fields when ``missing="fill"``.
    dtype : data-type, default=float
        Output dtype. With ``missing="skip"`` the data is parsed as float
        and cast after missing rows are removed.
    missing : {"fill", "skip"}, default="fill"
        "fill" replaces missing fields with ``missing_value``; "skip"
        drops every row containing a missing field.
    skip_header : int, default=0
        Number of leading lines (e.g. a header row) to ignore.

    Returns
    -------
    np.ndarray of shape (n_rows, n_columns)
        Always 2D, even for a single row or column. ``n_rows`` may be 0
        when every row is skipped or the file is empty.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    TypeError
        If path, delimiter or skip_header has the wrong type.
    ValueError
        If ``missing`` is unknown, skip_header is negative, or rows have
        inconsistent numbers of columns.

    Notes
    -----
    Time complexity: O(N) for N characters in the file.
    Space complexity: O(R * C) for the resulting array.

    Examples
    --------
    >>> data = read_csv("tests/sample.csv")
    >>> data.shape
    (2, 2)
    """
    skip_header = _validate_options(path, delimiter, missing, skip_header)

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()[skip_header:]

    lines = [line for line in lines if line.strip()]
    if not lines:
        return np.empty((0, 0), dtype=dtype)

    return _parse_lines(lines, delimiter, missing_value, dtype, missing)


def read_csv_chunks(path, chunk_size=1000, delimiter=",",
                    missing_value=np.nan, dtype=float, missing="fill",
                    skip_header=0):
    """
    Stream a CSV file as consecutive 2D arrays of at most ``chunk_size`` rows.

    The file is read lazily: each chunk is parsed only when requested,
    so memory use depends on ``chunk_size`` rather than on file size.

    Parameters
    ----------
    path : str or os.PathLike
        Path to the CSV file.
    chunk_size : int, default=1000
        Maximum number of input rows parsed per chunk (at least 1).
    delimiter, missing_value, dtype, missing, skip_header
        As for ``read_csv``.

    Yields
    ------
    np.ndarray of shape (n_chunk_rows, n_columns)
        2D chunk. With ``missing="skip"`` a chunk can have fewer than
        ``chunk_size`` rows, and chunks that become empty are not yielded.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (raised on the first ``next()``).
    TypeError
        If an argument has the wrong type.
    ValueError
        If chunk_size < 1, ``missing`` is unknown, or the number of
        columns changes between rows or chunks.

    Notes
    -----
    Time complexity: O(N) in total for N characters.
    Space complexity: O(chunk_size * C) per chunk.

    Examples
    --------
    >>> total = 0.0
    >>> for chunk in read_csv_chunks("tests/sample.csv", chunk_size=1):
    ...     total += chunk.sum()
    >>> total
    10.0
    """
    chunk_size = validate_int(chunk_size, "chunk_size", minimum=1)
    skip_header = _validate_options(path, delimiter, missing, skip_header)

    n_columns = None

    def emit(buffer):
        nonlocal n_columns
        chunk = _parse_lines(buffer, delimiter, missing_value, dtype, missing)

        if n_columns is None:
            n_columns = chunk.shape[1]
        elif chunk.shape[1] != n_columns:
            raise ValueError(
                f"Inconsistent number of columns: expected {n_columns}, "
                f"found {chunk.shape[1]}"
            )
        return chunk

    with open(path, "r", encoding="utf-8") as handle:
        for _ in range(skip_header):
            if not handle.readline():
                return

        buffer = []
        for line in handle:
            if not line.strip():
                continue
            buffer.append(line)

            if len(buffer) == chunk_size:
                chunk = emit(buffer)
                buffer = []
                if chunk.shape[0]:
                    yield chunk

        if buffer:
            chunk = emit(buffer)
            if chunk.shape[0]:
                yield chunk


__all__ = ["read_csv", "read_csv_chunks"]
