import matplotlib.pyplot as plt

acc = [96.3, 94.7, 93.9, 92.03333333, 90.96666667, 85.03333333]
err = [0.561248608, 2.25166605, 1.1414213562, 2.62741952, 3.382799629, 1.27410099]
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
