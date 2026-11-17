"""Tests for read_csv options and the streaming read_csv_chunks reader."""

import numpy as np
import pytest

from numcompute.io import read_csv, read_csv_chunks


@pytest.fixture
def csv_with_missing(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        "a,b,c\n"
        "1,2,3\n"
        "4,,6\n"
        "7,8,9\n"
        "\n"
        "10,11,\n"
        "13,14,15\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------
# read_csv
# ---------------------------------------------------------

def test_read_csv_fill_missing_with_nan(csv_with_missing):
    data = read_csv(csv_with_missing, skip_header=1)

    assert data.shape == (5, 3)
    assert np.isnan(data[1, 1]) and np.isnan(data[3, 2])


def test_read_csv_fill_missing_with_placeholder(csv_with_missing):
    data = read_csv(csv_with_missing, skip_header=1, missing_value=-1)
    assert data[1, 1] == -1 and data[3, 2] == -1


def test_read_csv_skip_missing_rows(csv_with_missing):
    data = read_csv(csv_with_missing, skip_header=1, missing="skip")
    assert np.array_equal(data, [[1, 2, 3], [7, 8, 9], [13, 14, 15]])


def test_read_csv_skip_then_cast_dtype(csv_with_missing):
    data = read_csv(csv_with_missing, skip_header=1, missing="skip", dtype=int)
    assert data.dtype.kind == "i"


def test_read_csv_tab_delimiter(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("1\t2\n3\t4\n", encoding="utf-8")
    assert np.array_equal(read_csv(path, delimiter="\t"), [[1, 2], [3, 4]])


def test_read_csv_is_always_2d(tmp_path):
    row = tmp_path / "row.csv"
    row.write_text("1,2,3\n", encoding="utf-8")
    column = tmp_path / "col.csv"
    column.write_text("1\n2\n3\n", encoding="utf-8")

    assert read_csv(row).shape == (1, 3)
    assert read_csv(column).shape == (3, 1)


def test_read_csv_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    assert read_csv(path).shape == (0, 0)


def test_read_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_csv(tmp_path / "nope.csv")


def test_read_csv_rejects_bad_options(csv_with_missing):
    with pytest.raises(ValueError, match="missing"):
        read_csv(csv_with_missing, missing="drop")
    with pytest.raises(ValueError):
        read_csv(csv_with_missing, skip_header=-1)
    with pytest.raises(TypeError):
        read_csv(123)


# ---------------------------------------------------------
# read_csv_chunks (streaming)
# ---------------------------------------------------------

@pytest.mark.parametrize("chunk_size", [1, 2, 3, 100])
def test_chunks_concatenate_to_full_read(csv_with_missing, chunk_size):
    chunks = list(read_csv_chunks(csv_with_missing, chunk_size=chunk_size,
                                  skip_header=1))
    full = read_csv(csv_with_missing, skip_header=1)

    assert all(chunk.ndim == 2 for chunk in chunks)
    assert all(chunk.shape[0] <= chunk_size for chunk in chunks)
    assert np.array_equal(np.vstack(chunks), full, equal_nan=True)


def test_chunk_sizes(csv_with_missing):
    sizes = [c.shape[0] for c in read_csv_chunks(csv_with_missing,
                                                 chunk_size=2, skip_header=1)]
    assert sizes == [2, 2, 1]


def test_chunks_skip_missing(csv_with_missing):
    chunks = list(read_csv_chunks(csv_with_missing, chunk_size=2,
                                  skip_header=1, missing="skip"))
    assert np.array_equal(
        np.vstack(chunks),
        read_csv(csv_with_missing, skip_header=1, missing="skip"),
    )
    assert all(chunk.shape[0] > 0 for chunk in chunks)


def test_chunks_are_lazy(csv_with_missing):
    iterator = read_csv_chunks(csv_with_missing, chunk_size=1, skip_header=1)
    first = next(iterator)
    assert np.array_equal(first, [[1, 2, 3]])


def test_chunks_support_streaming_aggregation(tmp_path):
    path = tmp_path / "big.csv"
    values = np.arange(1, 1001, dtype=float).reshape(-1, 2)
    np.savetxt(path, values, delimiter=",")

    total, count = 0.0, 0
    for chunk in read_csv_chunks(path, chunk_size=64):
        total += chunk.sum()
        count += chunk.shape[0]

    assert count == 500
    assert total == values.sum()


def test_chunks_reject_inconsistent_columns(tmp_path):
    path = tmp_path / "ragged.csv"
    path.write_text("1,2\n3,4\n5,6,7\n", encoding="utf-8")
    with pytest.raises(ValueError):
        list(read_csv_chunks(path, chunk_size=2))


def test_chunks_reject_invalid_chunk_size(csv_with_missing):
    with pytest.raises(ValueError):
        list(read_csv_chunks(csv_with_missing, chunk_size=0))
    with pytest.raises(TypeError):
        list(read_csv_chunks(csv_with_missing, chunk_size=2.5))


def test_chunks_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(read_csv_chunks(tmp_path / "nope.csv"))
