import time
import math

import numpy as np
from numpy.testing import assert_allclose
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import awkward as ak
import pandas as pd
import tensorflow as tf
import torch
import pytaco as pt
from pytaco import dense, compressed
import joblib

memory = joblib.Memory("joblib_cache", verbose=0)


def setup():
    rng = np.random.default_rng(123)
    ragged_data = []
    for i in range(1, 10_001):
        ragged_data.append(rng.random(size=i))
    ragged_data = np.asarray(ragged_data, dtype=object)
    regular_data = np.zeros((10_000, 10_000), dtype=np.float64)
    return ragged_data


def check_result(orig_data, result):
    if not isinstance(result, tf.RaggedTensor):
        assert len(result) == len(orig_data)
    else:
        assert result.shape[0] == len(orig_data)
    assert_allclose(result[10][1], orig_data[10][1] ** 4)


@memory.cache
def raw_python_bench(n_trials: int = 1):
    """
    Raw Python on the ragged NumPy array, with no zero-filling.
    """
    total_sec_l = []
    for trial in range(n_trials):
        ragged_data = setup()
        start = time.perf_counter()
        for row in range(len(ragged_data)):
            for col in range(len(ragged_data[row])):
                ragged_data[row][col] = math.sqrt(ragged_data[row][col])
        end = time.perf_counter()
        total_sec = end - start
        total_sec_l.append(total_sec)
    return total_sec_l, ragged_data


@memory.cache
def awkward_bench(n_trials: int = 1):
    """
    Using Awkward array to handle the compound mul calculation. The
    conversion to ak format is included in the timing.
    """
    total_sec_l = []
    granular_sec_l = []
    for trial in range(n_trials):
        ragged_data, regular_data = setup()
        start = time.perf_counter()
        ragged_data = ak.Array(ragged_data.tolist())
        granular_start = time.perf_counter()
        result = ragged_data * ragged_data * ragged_data * ragged_data
        granular_sec = time.perf_counter() - granular_start
        granular_sec_l.append(granular_sec)
        end = time.perf_counter()
        total_sec = end - start
        total_sec_l.append(total_sec)
    return total_sec_l, granular_sec_l, result


@memory.cache
def tf_bench(device, n_trials: int = 1):
    """
    Using tensorflow Ragged tensors for product op. Type/format
    conversions are included in the timing.
    """
    total_sec_l = []
    granular_sec_l = []
    for trial in range(n_trials):
        ragged_data = setup()
        start = time.perf_counter()
        with tf.device(device):
            ragged_data = tf.ragged.constant(ragged_data)
            granular_start = time.perf_counter()
            result = ragged_data * ragged_data * ragged_data * ragged_data
            # crude attempt to avoid potential lazy
            # eval issues (is that actual problem with tf?)
            # I didn't want to go as far as trying to force
            # eager evaluation globally though
            print(result[10][1])
            granular_sec = time.perf_counter() - granular_start
            granular_sec_l.append(granular_sec)
        end = time.perf_counter()
        total_sec = end - start
        total_sec_l.append(total_sec)
    return total_sec_l, granular_sec_l, result


@memory.cache
def torch_bench(device):
    """
    Using torch nested tensors for sqrt. Type/format
    conversions are included in the timing.
    """
    ragged_data = setup()
    start = time.perf_counter()
    with torch.device(device):
        ragged_data = torch.nested.nested_tensor(ragged_data.tolist())
        ragged_data = torch.sqrt(ragged_data)
    end = time.perf_counter()
    total_sec = end - start
    return [total_sec], ragged_data


@memory.cache
def pytaco_bench(n_trials: int = 1):
    total_sec_l = []
    granular_sec_l = []
    for trial in range(n_trials):
        ragged_data = setup()
        start = time.perf_counter()
        # effectively 0-fill to a sparse tensor:
        n = ragged_data.shape[0]
        # pytaco cannot accept the ragged Python object directly
        A = pt.tensor([n, n],
                      pt.format([dense, compressed]),
                      name="A",
                      dtype=pt.float64)
        # pay the cost to fill in the CSR-like array
        for row in range(len(ragged_data)):
            for col in range(len(ragged_data[row])):
                A.insert([row, col], ragged_data[row][col])
        result = pt.tensor([n, n],
                            pt.format([dense, compressed]),
                            name="result",
                            dtype=pt.float64)
        i, j = pt.get_index_vars(2)
        granular_start = time.perf_counter()
        result[i, j] = A[i, j] * A[i, j] * A[i, j] * A[i, j]
        result.evaluate()
        granular_sec = time.perf_counter() - granular_start
        granular_sec_l.append(granular_sec)
        result = result.to_array()
        end = time.perf_counter()
        total_sec = end - start
        total_sec_l.append(total_sec)
    return total_sec_l, granular_sec_l, result


def plot_results(bench_results):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4)
    colors = []
    for key in bench_results.keys():
        if "granular" in key:
            colors.append("orange")
        else:
            colors.append("blue")
    df = pd.DataFrame.from_dict(data=bench_results,
                                orient="columns")
    mean = df.mean()
    std = df.std()
    ax.bar(list(bench_results.keys()),
           height=mean,
           yerr=std,
           color=colors,
           log=True,
           capsize=8)
    ax.set_ylabel("Log of time (s)")
    ax.tick_params(axis='x', rotation=90)
    fig.tight_layout()
    fig.savefig("bench_compound_mul.png", dpi=300)


def main_bench():
    orig_data = setup()
    bench_results = {}
    #bench_results["Raw Python"], result = raw_python_bench(n_trials=3)
    #check_result(orig_data, result)
    bench_results["Awkward Array"], bench_results["Awkward Array\ngranular"], result = awkward_bench(n_trials=3)
    check_result(orig_data, result)
    bench_results["Tensorflow Ragged GPU"], bench_results["Tensorflow Ragged GPU\ngranular"], result = tf_bench(device="/device:GPU:0", n_trials=3)
    check_result(orig_data, result)
    bench_results["Tensorflow Ragged CPU"], bench_results["Tensorflow Ragged CPU\ngranular"], result = tf_bench(device="/device:CPU:0", n_trials=3)
    check_result(orig_data, result)
    bench_results["PyTaco"], bench_results["PyTaco\ngranular"], result = pytaco_bench(n_trials=3)
    check_result(orig_data, result)
    # NOTE: torch nested_tensor does not support sqrt op at this time
    #bench_results["torch_nested_cpu"], result = torch_bench(device="cpu")
    #check_result(orig_data, result)
    #bench_results["torch_nested_gpu"], result = torch_bench(device="mps")
    #check_result(orig_data, result)
    plot_results(bench_results)


if __name__ == "__main__":
    main_bench()
