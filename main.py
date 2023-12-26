from train import run_train_process
from evaluation import run_evaluation_process
import json

with open("config.json", "r") as config_file:
    config = json.load(config_file)

data_path_main = config["data_path_main"]


if __name__ == "__main__":
    run_train_process(data_path_main)
    run_evaluation_process(data_path_main)
