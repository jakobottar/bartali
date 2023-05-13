import glob
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


def create_trainval_file(rdir, fold_num, mags, trainval_file_dir):
    """
    Generate train/val text file of a dataset with a specified kth fold
    """

    total_fold = 5

    train_files = []
    val_files = []
    for r in ROUTES:
        min_files = 10000
        total_files = {}
        ## Get filenames across specified magnifications ##
        for m in mags:
            curr_key = f"{r}_{m}"

            curr_dir = f"{rdir}/{r}/{m}"
            curr_files = glob.glob(curr_dir + "/*")
            curr_files.sort()
            total_files[curr_key] = curr_files
            if len(curr_files) < min_files:
                min_files = len(curr_files)

        curr_routes = []
        for i in range(min_files):
            curr_val = total_files[f"{r}_{mags[0]}"][i]
            for m in mags[1:]:
                curr_val += f",{total_files[f'{r}_{m}'][i]}"
            curr_val += f",{r}\n"
            curr_routes.append(curr_val)

        num_per_img = 4
        ind_split = np.array_split(
            np.arange(len(curr_routes) // num_per_img), total_fold
        )

        ## Split into train/val based on specified fold ##
        curr_val_files = [
            curr_routes[i]
            for i in range(
                ind_split[fold_num][0] * num_per_img,
                ind_split[fold_num][-1] * num_per_img,
            )
        ]
        curr_train_files = [x for x in curr_routes if x not in curr_val_files]
        train_files += curr_train_files
        val_files += curr_val_files

    random.shuffle(train_files)

    write_to_files(
        train_files,
        f"{trainval_file_dir}/train_{fold_num}.csv",
        ["10000x", "25000x", "50000x", "100000x", "label"],
    )
    write_to_files(
        val_files,
        f"{trainval_file_dir}/val_{fold_num}.csv",
        ["10000x", "25000x", "50000x", "100000x", "label"],
    )


def write_to_files(files, filename, column_names):
    """
    Parse to text file
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(",".join(column_names) + "\n")
        for l in files:
            f.write(l)


if __name__ == "__main__":
    for fold_num in range(0, 5):
        create_trainval_file(
            "/scratch_nvme/jakobj/multimag",
            fold_num,
            ["10000x", "25000x", "50000x", "100000x"],
            ".",
        )
