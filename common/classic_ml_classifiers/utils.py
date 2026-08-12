import numpy as np
import random


def prepare_Xs_and_Ys(logger, calc_embeddings_fn, split, always_calc_embeddings, positive_fasta_file,
                      negative_fasta_file, embeddings_dir):
    Xs_positive, Xs_negative = calc_embeddings_fn(split, always_calc_embeddings=always_calc_embeddings,
                                                   positive_fasta_file=positive_fasta_file,
                                                   negative_fasta_file=negative_fasta_file,
                                                   embeddings_dir=embeddings_dir)

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    # Shuffle
    combined = list(zip(Xs, Ys))
    random.shuffle(combined)
    shuffled_Xs, shuffled_Ys = zip(*combined)
    shuffled_Xs = np.array(shuffled_Xs)

    logger.info(f"Loaded {split} data: Xs_{split}.shape = {Xs.shape}, Ys_{split}.shape = {len(Ys)}")

    return shuffled_Xs, shuffled_Ys
