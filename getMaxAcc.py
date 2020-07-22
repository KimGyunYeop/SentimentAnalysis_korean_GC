import pandas as pd
import os
import argparse

args = argparse.ArgumentParser()

args.add_argument("--result_dir", type=str, required=True)

args = args.parse_args()

result_path = os.path.join("ckpt", args.result_dir, "test")
epoch_list = os.listdir(result_path)

acc_dict = dict()
for i in epoch_list:
    with open(os.path.join(result_path,epoch_list),"r") as fp:
        acc_dict["i"] = float(fp.readline())

print(max(acc_dict.values()))