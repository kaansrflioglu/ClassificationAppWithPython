from train import run_train_process
from evaluation import run_evaluation_process

data_path_main = "dataset_duzcetip"

if __name__ == "__main__":
    run_train_process(data_path_main)
    run_evaluation_process(data_path_main)
