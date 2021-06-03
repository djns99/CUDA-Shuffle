import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import subprocess
from io import StringIO
import scipy.stats

matplotlib.use('Agg')
sns.set()
matplotlib.rcParams['text.usetex'] = True
plt.rc('font', family='serif')
sns.set_style("whitegrid")

num_samples = 1000000
output = subprocess.run(
    ["./build/test/MMD", str(num_samples)], capture_output=True)
mmd_df = pd.read_csv(StringIO(output.stdout.decode("utf-8")))
output = subprocess.run(
    ["./build/test/ChiSquared", str(num_samples)], capture_output=True)
chi_df = pd.read_csv(StringIO(output.stdout.decode("utf-8")))
# Add p value
sig1 = scipy.stats.chi2.ppf(1-0.01, df=119)
for r in range(chi_df["Rounds"].min(), chi_df["Rounds"].max()+1):
    chi_df = chi_df.append({"Algorithm": "$\\alpha=0.01$",
                            "$\chi^2$": sig1, "Rounds": r}, ignore_index=True)

commbined = mmd_df["Algorithm"].append(chi_df["Algorithm"])
unique = commbined.unique()
palette = dict(zip(unique, sns.color_palette(n_colors=len(unique))))

for n in mmd_df["n"].unique():
    plt.figure(figsize=(6, 4.5))
    algs = mmd_df["Algorithm"].unique()
    sns.lineplot(data=mmd_df.loc[mmd_df['n'] == n], x="Rounds",
                 y="$|\hat{\mathrm{MMD}}^2|$", hue="Algorithm", style="Algorithm", palette=palette)
    plt.yscale("log")
    plt.savefig("MMD_n{}.png".format(n), bbox_inches="tight", dpi=1000)
    plt.clf()

plt.figure(figsize=(6, 4.5))
sns.lineplot(data=chi_df, x="Rounds", y="$\chi^2$",
             hue="Algorithm", style="Algorithm", palette=palette)
plt.yscale("log")
plt.savefig("ChiSquared.png", bbox_inches="tight", dpi=1000)
