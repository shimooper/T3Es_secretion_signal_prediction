import subprocess
import argparse
from multiprocessing import Pool
from src.utils.consts import MODEL_ID_TO_MODEL_NAME


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', help='The number of cpus to use', type=int, required=True)

    return parser.parse_args()


def run_classification_with_classic_ML(model_id, n_jobs):
    cmd = f"python classification_with_classic_ML.py --model_id {model_id} --n_jobs {n_jobs}"
    subprocess.run(cmd, shell=True, check=True)


def run_esm_classification_fine_tune(model_id):
    cmd = f"python esm_classification_fine_tune.py --model_id {model_id}"
    subprocess.run(cmd, shell=True, check=True)


def main():
    args = get_arguments()

    esm_model_ids = list(MODEL_ID_TO_MODEL_NAME.keys())[:6]

    # for model_id in esm_model_ids:
    #     run_classification_with_classic_ML(model_id, args.cpus)

    with Pool(processes=args.cpus) as pool:
        pool.map(run_esm_classification_fine_tune, esm_model_ids)


if __name__ == "__main__":
    main()
