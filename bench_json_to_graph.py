import json
import matplotlib
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
import pandas as pd
from collections import OrderedDict

plt.style.use("seaborn")
matplotlib.rcParams['text.usetex'] = True
plt.rc('font', family='serif')


def is_mean(entry):
    return "name" in entry and "_mean" == entry["name"][-5:]


def results_to_csv(results):
    csv_list = [["size"]]
    sizes = []
    for name in sorted(results.keys()):
        csv_list[0].append(name)
        sizes = sorted(results[name].keys())

    for i, size in enumerate(sizes):
        csv_list.append([str(size)])
        for name in sorted(results.keys()):
            csv_list[i + 1].append(str(results[name][size]))
    with open(sys.argv[2], "w") as out_file:
        out_file.write("\n".join([",".join(x) for x in csv_list]))

to_keep = None
# to_keep = [("FeistelBijectiveScanShuffle Non-Power of Two", "Feistel"),
#            ("LCGBijectiveScanShuffle Non-Power of Two", "LCG"),
#            ("MergeShuffle Non-Power of Two", "Merge"),
#            ("RaoSandeliusShuffle Non-Power of Two", "RS"),
#            ("FisherYatesShuffle Non-Power of Two", "FY"),
#            ("GatherShuffle Non-Power of Two", "Gather"),
#            ("ScatterShuffle Non-Power of Two", "Scatter")]
use_power_two = False


def filter_out_unwanted_algorithms(results):
    output = OrderedDict()
    if to_keep is not None:
        for keep in to_keep:
            output[keep[1]] = results[keep[0]]
    else:
        for key in results.keys():
            if "Non-Power of Two" in key:
                if not use_power_two:
                    output[key] = results[key]
            elif use_power_two:
                output[key] = results[key]

    return output

# Functions for name conversion
def parse_dart_throwing(name):
    if "HostDartThrowing" not in name:
        return "GPU Dart Throwing 4x Buffer"
    if "5, 4" in name:
        return "Host Dart Throwing 1.25x Buffer"
    if "3, 2" in name:
        return "Host Dart Throwing 1.5x Buffer"
    if "2, 1" in name:
        return "Host Dart Throwing 2x Buffer"
    return "Host Dart Throwing 4x Buffer"


def parse_scatter_gather(name):
    if name == "benchmarkScatterGather<ScatterShuffle<thrust::host_vector<DataType>, DefaultRandomGenerator, false>>":
        return "Single Threaded Host Scatter"
    if name == "benchmarkScatterGather<GatherShuffle<thrust::host_vector<DataType>, DefaultRandomGenerator, false>>":
        return "Single Threaded Host Gather"
    if name == "benchmarkScatterGather<ScatterShuffle<thrust::host_vector<DataType>>>":
        return "Parallel Host Scatter"
    if name == "benchmarkScatterGather<GatherShuffle<thrust::host_vector<DataType>>>":
        return "Parallel Host Gather"
    if name == "benchmarkScatterGather<ScatterShuffle<thrust::device_vector<DataType>>>":
        return "GPU Scatter"
    if name == "benchmarkScatterGather<GatherShuffle<thrust::device_vector<DataType>>>":
        return "GPU Gather"
    return None


def get_host_device_prefix(name):
    if "TBB" in name:
        return "TBB"
    return "GPU"


def get_rounds(name):
    if "target_num_rounds" in name:
        return "16"
    string = "WyHashRoundFunction<"
    start = name.index(string) + len(string)
    end = name.index("ul", start)
    return name[start:end]


def get_fast_gen(name):
    if ", true" in name:
        return "Raw"
    return ""


def get_rng_name(name, prefix):
    return get_host_device_prefix(name) + " " + prefix + " " + get_fast_gen(name) + " 16 rounds"


def convert_name(name):
    single_names = ["StdShuffle", "MergeShuffle", "RaoSandeliusShuffle", "Philox"]
    for part in single_names:
        if "benchmarkFunction<" + part in name:
            return part
    if "DartThrowing<" in name:
        return parse_dart_throwing(name)
    if "benchmarkScatterGather" in name:
        return parse_scatter_gather(name)
    if "WyHash" in name:
        return get_host_device_prefix(name) + " WyHash " + get_rounds(name) + " rounds"
    if "RC5" in name:
        return get_host_device_prefix(name) + " RC5 16 rounds"
    if "LCGBijectiveScanShuffle" in name:
        if "tbb::vector" in name:
            return "TBB LCG Bijection"
        return "GPU LCG Bijection"
    if "Taus88Ranlux" in name:
        return get_rng_name(name, "Taus88 + Ranlux")
    if "Taus88LCG" in name:
        return get_rng_name(name, "Taus88 + LCG")
    if "RanluxLCG" in name:
        return get_rng_name(name, "Ranlux + LCG")

    return None


with open(sys.argv[1], "r") as read_file:
    plt.figure(figsize=(6, 4.5))
    data = json.load(read_file)
    func = lambda entry: is_mean(entry)
    means = list(filter(func, data["benchmarks"]))
    results = OrderedDict()

    for mean in means:
        split_name = mean["run_name"].split("/")
        test_name = convert_name(split_name[0])
        run_name = test_name + (
            " Power of Two" if split_name[2] == "0" else " Non-Power of Two")
        size = 2**int(split_name[1]) + int(split_name[2])
        if run_name not in results:
            results[run_name] = {}
        assert size not in results[run_name]
        results[run_name][size] = mean["items_per_second"] / 1e6
    # print(json.dumps(results, indent=4, sort_keys=True))
    results = filter_out_unwanted_algorithms(results)
    index = sorted(next(iter(results.values())).keys())

    df = pd.DataFrame(index=index)
    for test_name in results.keys():
        test_res = OrderedDict(sorted(results[test_name].items()))
        df[test_name] = test_res.values()
        # Check all columns have the same sizes
        print(index)
        print(results[test_name].items())
        assert np.array_equal(index, sorted(test_res.keys())), (index, test_res.keys())

    # TODO Tune this
    # dash_styles = ["",
    #                (4, 1.5),
    #                (1, 1),
    #                (3, 1, 1.5, 1),
    #                (5, 1, 1, 1),
    #                (5, 1, 2, 1, 2, 1),
    #                (2, 2, 3, 1.5),
    #                (1, 2.5, 3, 1.2)]
    ax = sns.lineplot(data=df, markers=True)
    plt.xlabel("Input Size")
    plt.ylabel("Throughput (millions item/s)")
    plt.xscale("log", basex=2)
    plt.yscale("log")
    plt.legend()
    # plt.title("Shuffle Algorithm Performance")
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(sys.argv[2], orientation='landscape', bbox_inches="tight", papertype="a0",
                quality=100, dpi=1000)
    # Format float strings
    df = df.applymap(
        lambda x: np.format_float_positional(x, precision=4, fractional=False, trim="-"))

    df.index = ["$2^{" + str(int(np.log2(m))) + "} + 1$" for m in df.index]
    df.index.name = "Input size"

    print(df.to_latex(float_format="%.2f", column_format="r" * (len(df) + 1), escape=False))
