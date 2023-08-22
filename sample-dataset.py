"""
generate sample dataset from training/testing set for publishing
"""
import json
import os

import numpy as np
import skimage

ROOT = "/scratch_nvme/jakobj/multimag/"

ROUTES = [
    # "U3O8ADU",
    "U3O8AUC",
    "U3O8MDU",
    "U3O8SDU",
    "UO2ADU",
    "UO2AUCd",
    "UO2AUCi",
    "UO2SDU",
    "UO3ADU",
    "UO3AUC",
    "UO3MDU",
    "UO3SDU",
]

DETECTORS = ["Nova"]

MODES = ["SE"]

MAGS = ["10000x", "25000x", "50000x", "100000x"]

NUM_IMGS = 4


if __name__ == "__main__":
    with open(
        "/scratch_nvme/jakobj/multimag/nova_train_0.json", "r", encoding="utf-8"
    ) as f:
        full_dataset = json.load(f)
    count = 0
    new_dataset = []
    for route in ROUTES:
        subset = [x for x in full_dataset if x["route"] == route][:NUM_IMGS]
        for sample in subset:
            data = {"route": route, "detector": "Nova Nano", "mode": "SE", "files": []}
            print(sample["files"])
            for file, mag in zip(sample["files"], MAGS):
                image = skimage.io.imread(file, as_gray=False)
                skimage.io.imsave(f"./sample-dataset/img_{count}_{mag}.png", image)

                data["files"].append(f"img_{count}_{mag}.png")

            new_dataset.append(data)

            count += 1

    with open("./sample-dataset/dataset.json", "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=4)
