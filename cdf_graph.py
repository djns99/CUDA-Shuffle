import matplotlib.pyplot as plt
import sys
from scipy.stats import chi2, norm
import numpy as np

fig_count = 0

def split_name(line):
    split = line.rsplit(':', 1)
    return split[0], split[1]

def plot_chi2(line):
    global fig_count
    name, line = split_name(line)
    split_line = line.split(',')
    df = int(split_line[1])
    nums = [float(x) for x in split_line[2:] if x.strip()]
    fig, ax = plt.subplots(1, 1)
    x = np.linspace(chi2.ppf(0.0001, df=df),
                    chi2.ppf(0.9999, df=df), 10000)
    ax.plot(x, chi2.pdf(x, df=df),
            'k-', lw=1, label='chi2 pdf')

    ax.hist(nums, density=True, histtype='stepfilled', bins=50, alpha=0.2)
    ax.legend(loc='best', frameon=False)
    ax.set_title(name, fontsize=6, wrap=True)
    plt.savefig("figure" + str(fig_count) + ".png")
    fig_count += 1
    plt.show()


def plot_normal(line):
    global fig_count
    name, line = split_name(line)
    split_line = line.split(',')
    mean = float(split_line[1])
    stddev = float(split_line[2])
    nums = [int(x) for x in split_line[3:] if x.strip()]

    fig, ax = plt.subplots(1, 1)
    x = np.linspace(norm.ppf(0.0001, loc=mean, scale=stddev),
                    norm.ppf(0.9999, loc=mean, scale=stddev), 10000)
    ax.plot(x, norm.pdf(x, loc=mean, scale=stddev),
            'k-', lw=1, label='norm pdf')

    ax.hist(nums, density=True, histtype='stepfilled', bins=50, alpha=0.2)
    ax.legend(loc='best', frameon=False)
    ax.set_title(name, fontsize=6, wrap=True)
    plt.savefig("figure" + str(fig_count) + ".png")
    fig_count += 1
    plt.show()


while True:
    line = sys.stdin.readline()
    if not line or line is None:
        break
    line = line.strip()
    use_chi2 = False

    if "chi2" in line:
        plot_chi2(line)
    elif "normal" in line:
        plot_normal(line)