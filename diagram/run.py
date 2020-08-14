from model.random_guess import *
import argparse


def run(data_path):
    data_path = "dataset"
    
    private_data = get_splited_private_tasks(data_path)
    
    total_ans_list = random_guess(private_data,data_path)

    generate_ans_file(total_ans_list)


def parse_args():
    parser = argparse.ArgumentParser(description='Basic setting')

    """set path of dataset"""
    parser.add_argument('--path',dest="DATASET_PATH",
                      help='{path of dataset}',
                      default='dataset',
                      type=str, required=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    path = args.DATASET_PATH
    run(path)