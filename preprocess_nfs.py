import numpy as np
import glob
import random

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


def create_trainval_file(
    rdir, fold_num, mags, trainval_file_dir, trainval_file_corename
):
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
            curr_key = "%s_%s" % (r, m)

            curr_dir = "%s/%s/%s" % (rdir, r, m)
            curr_files = glob.glob(curr_dir + "/*")
            curr_files.sort()
            total_files[curr_key] = curr_files
            if len(curr_files) < min_files:
                min_files = len(curr_files)

        curr_routes = []
        for i in range(min_files):
            curr_val = total_files["%s_%s" % (r, mags[0])][i]
            for m in mags[1:]:
                curr_val += " %s" % total_files["%s_%s" % (r, m)][i]
            curr_val += " %s\n" % r
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
        "%s/train_%s_fold%d.txt"
        % (trainval_file_dir, trainval_file_corename, fold_num),
    )
    write_to_files(
        val_files,
        "%s/val_%s_fold%d.txt" % (trainval_file_dir, trainval_file_corename, fold_num),
    )


def write_to_files(files, filename):
    """
    Parse to text file
    """

    f = open(filename, "w")
    for l in files:
        f.write(l)

    f.close()


if __name__ == "__main__":
    create_trainval_file(
        "/scratch/jakobj/multimag",
        0,
        ["10000x", "25000x", "50000x", "100000x"],
        ".",
        "data",
    )
