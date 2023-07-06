import matplotlib.pyplot as plt

acc = [96.3, 96.40666667, 94.57666667, 94.2, 93.93333333, 87.76666667]
err = [0.561248608, 0.7736493607, 1.320845689, 0.7549834435, 0.7571877794, 3.121431296]
dropped = [0, 2, 4, 6, 8, 10]

plt.errorbar(dropped, acc, yerr=err, fmt="none", color="black", capsize=5)
plt.plot(dropped, acc, "o", color="#CC0000")
plt.xlabel("Number of Classes Dropped")
plt.ylabel("Validation Accuracy (%)")
plt.ylim(75, 100)


fig = plt.gcf()
fig.set_size_inches(4.3, 2.7)
fig.tight_layout()

fig.savefig("./drop-n-accuracy.png", dpi=300, pad_inches=1)
