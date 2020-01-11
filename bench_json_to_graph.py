import json
import matplotlib
import matplotlib.pyplot as plt
import sys
from labellines import labelLine, labelLines
import numpy as np

def is_mean( entry ):
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
            csv_list[i+1].append(str(results[name][size]))
    with open(sys.argv[2], "w") as out_file:
        out_file.write("\n".join([",".join(x) for x in csv_list]))

to_keep = [("FeistelBijectiveScan", "Feistel"), ("LCGBijectiveScan", "LCG"), ("MergeShuffle", "Merge"), ("RaoSandelius", "Rao-Sandelius"), ("FisherYates", "Fisher-Yates"), ("Gather", "Random Memory Access")]
def filter_out_unwanted_algorithms( results ):
    output = {}
    for key in results.keys():
        found = False
        for keep in to_keep:
            if keep[0] in key and "Non-Power" in key:
                found=True
                break
        if found:
            output[keep[1]] = results[key]
    return output



with open(sys.argv[1], "r") as read_file:
    data = json.load(read_file)
    func = lambda entry : is_mean(entry)
    means = list(filter( func, data["benchmarks"] ))
    results = {}

    for mean in means:
        split_name = mean["run_name"].split("/")
        run_name = split_name[0].split("<")[1] + (" Power of Two" if split_name[2] == "0" else " Non-Power of Two")
        size = int(split_name[1])
        if run_name not in results:
            results[run_name] = {}
        results[run_name][size] = mean["items_per_second"] / 1e6
    # print(json.dumps(results, indent=4, sort_keys=True))
    results = filter_out_unwanted_algorithms(results)
    for test_name in results.keys():
        test_res = results[test_name]
        list1 = []
        list2 = []
        for size in sorted(test_res.keys()):
            list1.append(size)
            list2.append(test_res[size])
        plt.plot(list1, list2, label=test_name, linewidth=2)
    plt.xlabel("Input Size")
    plt.ylabel("Throughput (millions item/s)")
    plt.xscale("log", basex=2)
    plt.yscale("log")
    # plt.legend(frameon=False, shadow=False, framealpha=0)
    plt.title("Shuffle Algorithm Performance")
    # labelLines(plt.gca().get_lines(), align=False, zorder=10, xvals=[2**13, 2**20, 2**24, 2**17, 2**13, 2**15], bbox=dict(alpha=0), color="black")
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(sys.argv[2], transparent=True, orientation='landscape',bbox_inches="tight", papertype="a0", quality=100, dpi=1000)
