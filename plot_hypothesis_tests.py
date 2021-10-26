import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import subprocess
from io import StringIO
import scipy.stats
from scipy import special

matplotlib.use("Agg")
sns.set()
matplotlib.rcParams["text.usetex"] = True
plt.rc("font", family="serif")
sns.set_style("whitegrid")


def expected_value(n, l):
    mgf_expected = 1.0
    norm = (n * (n - 1)) / 2
    for j in range(1, n + 1):
        mgf_expected *= (1 - np.exp(j * (-l / norm))) / (j * (1 - np.exp(-l / norm)))
    return mgf_expected


def normal_acceptance(alpha, n, num_samples, l):
    var = expected_value(n, 2 * l) - np.power(expected_value(n, l), 2.0)
    mmd_var = (2.0 * var) / num_samples
    return np.sqrt(2 * mmd_var) * special.erfinv(1 - alpha)


def hoeffding_acceptance(alpha, num_samples):
    return np.sqrt(np.log(2.0 / alpha) / num_samples)


num_samples = 100000
output = subprocess.run(["./build/test/MMD", str(num_samples)], capture_output=True)
mmd_df = pd.read_csv(StringIO(output.stdout.decode("utf-8")))
mmd_df.to_csv("mmd_results.csv")
output = subprocess.run(
    ["./build/test/ChiSquared", str(num_samples)], capture_output=True
)
chi_df = pd.read_csv(StringIO(output.stdout.decode("utf-8")))
# Add p value
chi_alpha = 0.05
sig1 = scipy.stats.chi2.ppf(1 - chi_alpha, df=119)
for r in range(chi_df["Rounds"].min(), chi_df["Rounds"].max() + 1):
    chi_df = chi_df.append(
        {"Algorithm": "$\\alpha=0.05$", "$\chi^2$": sig1, "Rounds": r},
        ignore_index=True,
    )

# add acceptance thresholds
for n in mmd_df["n"].unique():
    for rounds in mmd_df["Rounds"].unique():
        mmd_df = mmd_df.append(
            {
                "Rounds": rounds,
                "n": n,
                "Algorithm": "$\\alpha_N=0.05$",
                "$|\hat{\mathrm{MMD}}^2|$": normal_acceptance(
                    0.05, n, num_samples, l=5.0
                ),
            },
            ignore_index=True,
        )
        mmd_df = mmd_df.append(
            {
                "Rounds": rounds,
                "n": n,
                "Algorithm": "$\\alpha_H=0.05$",
                "$|\hat{\mathrm{MMD}}^2|$": hoeffding_acceptance(0.05, num_samples),
            },
            ignore_index=True,
        )

commbined = mmd_df["Algorithm"].append(chi_df["Algorithm"])
unique = commbined.unique()
palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))


for n in mmd_df["n"].unique():
    plt.figure(figsize=(6, 4.5))
    algs = mmd_df["Algorithm"].unique()
    sns.lineplot(
        data=mmd_df.loc[mmd_df["n"] == n],
        x="Rounds",
        y="$|\hat{\mathrm{MMD}}^2|$",
        hue="Algorithm",
        style="Algorithm",
        palette=palette,
    )
    plt.yscale("log")
    plt.savefig("MMD_n{}.png".format(n), bbox_inches="tight", dpi=1000)
    plt.clf()

plt.figure(figsize=(6, 4.5))
sns.lineplot(
    data=chi_df,
    x="Rounds",
    y="$\chi^2$",
    hue="Algorithm",
    style="Algorithm",
    palette=palette,
)
plt.yscale("log")
plt.savefig("ChiSquared.png", bbox_inches="tight", dpi=1000)
