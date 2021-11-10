import json
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
import pandas as pd
from collections import OrderedDict

matplotlib.use("Agg")
sns.set()
matplotlib.rcParams["text.usetex"] = True
plt.rc("font", family="serif")


def is_mean(entry):
    return "name" in entry and "_mean" == entry["name"][-5:]


def filter_out_unwanted_algorithms(results, key):
    output = OrderedDict()
    for k, v in key.items():
        output[v] = results[k]
    return output


def get_data_as_dict():
    with open(sys.argv[1], "r") as read_file:
        data = json.load(read_file)
        means = list(filter(is_mean, data["benchmarks"]))
        results = OrderedDict()

        for mean in means:
            split_name = mean["run_name"].split("/")
            run_name = split_name[0] + split_name[2]
            size = 2 ** int(split_name[1]) + int(split_name[2])
            if run_name not in results:
                results[run_name] = {}
            assert size not in results[run_name]
            results[run_name][size] = (
                mean["items_per_second"] / 1e6,
                mean["real_time"] / 1e9,
            )
    return results


def plot_algorithms(name, results, key, colour_map):
    filtered_results = filter_out_unwanted_algorithms(results, key)
    index = sorted(next(iter(filtered_results.values())).keys())
    df = pd.DataFrame(index=index)
    for test_name in filtered_results.keys():
        test_res = OrderedDict(sorted(filtered_results[test_name].items()))
        df[test_name] = list(zip(*test_res.values()))[0]

    colours = [colour_map[k] for k in key.keys()]
    sns.set_palette(colours)
    plt.figure(figsize=(6, 4.5))
    sns.lineplot(data=df, markers=True)
    plt.xlabel("Input Size")
    plt.ylabel("Throughput (millions item/s)")
    plt.xscale("log", basex=2)
    plt.yscale("log")
    plt.legend()
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(name + ".png", orientation="landscape", bbox_inches="tight", dpi=1000)

    df.index = ["$2^{" + str(int(np.log2(m))) + "} + 1$" for m in df.index]
    df.index.name = "Input size"

    print(
        df.to_latex(
            float_format="%.2f", column_format="r" * (len(df) + 1), escape=False
        )
    )

    runtime_df = pd.DataFrame(index=index)
    for test_name in filtered_results.keys():
        test_res = OrderedDict(sorted(filtered_results[test_name].items()))
        runtime_df[test_name] = list(zip(*test_res.values()))[1]
    print(runtime_df)
    plt.clf()
    plt.figure(figsize=(6, 4.5))
    sns.lineplot(data=runtime_df, markers=True)
    plt.xlabel("Input Size")
    plt.ylabel("Runtime (s)")
    plt.xscale("log", basex=2)
    plt.yscale("log")
    plt.legend()
    ax = plt.subplot(111)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.savefig(
        name + "_runtime.png", orientation="landscape", bbox_inches="tight", dpi=1000
    )


def main():
    results = get_data_as_dict()
    experiments = {
        "GPUShuffle": {
            "benchmarkScatterGather<GatherShuffle<thrust::device_vector"
            "<DataType>>>1": "Gather",
            "benchmarkFunction<PhiloxBijectiveScanShuffle<>>1": "VarPhilox",
            "benchmarkFunction<LCGBijectiveScanShuffle<thrust::device_vector<DataType>>>1": "LCG",
            "benchmarkFunction<DartThrowing<thrust::device_vector<DataType"
            ">>>1": "DartThrowing",
            "benchmarkFunction<SortShuffle<thrust::device_vector<DataType"
            ">>>1": "SortShuffle",
        },
        "BijectiveComparison": {
            "benchmarkScatterGather<GatherShuffle<thrust::device_vector<DataType>>>1": "Gather",
            "benchmarkFunction<BasicPhiloxBijectiveScanShuffle<>>1": "Bijective0",
            "benchmarkFunction<TwoPassPhiloxBijectiveScanShuffle<>>1": "Bijective1",
            "benchmarkFunction<PhiloxBijectiveScanShuffle<>>1": "Bijective2",
            "benchmarkFunction<PhiloxBijectiveScanShuffle<>>0": "Bijective2(n=m)",
        },
        "CPUShuffle": {
            "benchmarkScatterGather<GatherShuffle<thrust::host_vector<DataType>>>1": "Gather",
            "benchmarkFunction<PhiloxBijectiveScanShuffle<thrust::tbb::vector<DataType"
            ">>>1": "VarPhilox",
            "benchmarkFunction<HostDartThrowing<std::vector<DataType>>>1": "DartThrowing",
            "benchmarkFunction<StdShuffle<std::vector<DataType>>>1": "std::shuffle",
            "benchmarkFunction<RaoSandeliusShuffle<std::vector<DataType>>>1": "RS",
            "benchmarkFunction<MergeShuffle<std::vector<DataType>>>1": "MergeShuffle",
            "benchmarkFunction<SortShuffle<thrust::host_vector<DataType>>>1": "SortShuffle",
        },
    }

    unique_algorithms = [x for v in experiments.values() for x in v.keys()]
    rs = np.random.RandomState(235)
    palette = sns.color_palette("tab20", n_colors=len(unique_algorithms))
    rs.shuffle(palette)
    colours = OrderedDict((alg, col) for (alg, col) in zip(unique_algorithms, palette))
    sns.set_style("whitegrid")
    for name, keys in experiments.items():
        plot_algorithms(name, results, keys, colours)


if __name__ == "__main__":
    main()
