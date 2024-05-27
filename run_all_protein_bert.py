import subprocess
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpus', help='The number of cpus to use', type=int, required=True)

    return parser.parse_args()


def run_classification_with_classic_ML(model_id, n_jobs):
    cmd = f"python classification_with_classic_ML.py --model_id {model_id} --n_jobs {n_jobs}"
    subprocess.run(cmd, shell=True, check=True)


def run_protein_bert_classification_fine_tune():
    cmd = "python protein_bert_classification_fine_tune.py"
    subprocess.run(cmd, shell=True, check=True)


def main():
    args = get_arguments()

    protein_bert_model_id = "protein_bert"

    # run_classification_with_classic_ML(protein_bert_model_id, args.cpus)
    run_protein_bert_classification_fine_tune()


if __name__ == "__main__":
    main()
