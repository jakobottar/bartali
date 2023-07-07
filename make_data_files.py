import glob
import json
import random

import numpy as np

ROUTES = [
    "U3O8ADU",
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

SEMS = ["Nova", "Helios", "Teneo"]
# SEMS = ["Nova"]

DETECTORS = ["SE", "BSE"]
# DETECTORS = ["SE"]

MAGS = ["10000x", "25000x", "50000x", "100000x"]


def create_trainval_file(rdir, fold_num):
    """
    Generate train/val text file of a dataset with a specified kth fold
    """
    TOTAL_FOLD = 5
    NUM_PER_IMG = 4

    train_files = []
    val_files = []
    for route in ROUTES:
        total_files = {}
        # min_files = 10_000
        for sem in SEMS:
            for detector in DETECTORS:
                for mag in MAGS:
                    curr_key = f"{route}_{sem}_{detector}_{mag}"

                    curr_dir = f"{rdir}/{route}/{sem}/{detector}/{mag}"
                    curr_files = glob.glob(curr_dir + "/*")
                    curr_files.sort()
                    total_files[curr_key] = curr_files
                    # min_files = min(len(curr_files), min_files)

        curr_routes = []
        for i in range(360):
            curr_val = {"route": route, "files": []}
            for sem in SEMS:
                for detector in DETECTORS:
                    for mag in MAGS:
                        try:
                            # print(f"{route}_{sem}_{detector}_{mag}")
                            curr_val["files"].append(
                                f"{total_files[f'{route}_{sem}_{detector}_{mag}'][i]}"
                            )
                        except IndexError:
                            continue

            if len(curr_val["files"]) > 0:
                curr_routes.append(curr_val)

        splits = np.arange(len(curr_routes) // NUM_PER_IMG)
        random.shuffle(splits)
        ind_split = np.array_split(splits, TOTAL_FOLD)

        print(ind_split)

        # Split into train/val based on specified fold
        curr_val_files = [
            curr_routes[i]
            for i in range(
                ind_split[fold_num][0] * NUM_PER_IMG,
                ind_split[fold_num][-1] * NUM_PER_IMG,
            )
        ]
        curr_train_files = [x for x in curr_routes if x not in curr_val_files]
        train_files += curr_train_files
        val_files += curr_val_files

    with open(f"full_train_{fold_num}.json", "w", encoding="utf-8") as f:
        json.dump(train_files, f, ensure_ascii=False, indent=4)

    with open(f"full_val_{fold_num}.json", "w", encoding="utf-8") as f:
        json.dump(val_files, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    for fold_num in range(0, 5):
        create_trainval_file(
            "/scratch_nvme/jakobj/multimag",
            fold_num,
        )
