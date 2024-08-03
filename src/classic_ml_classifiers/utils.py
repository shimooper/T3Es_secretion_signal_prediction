import numpy as np
import random


def prepare_Xs_and_Ys(logger, model_id, split, always_calc_embeddings):
    if model_id == 'protein_bert':
        from src.pretrained_embeddings.calc_proteinbert_embeddings import calc_embeddings
        Xs_positive, Xs_negative = calc_embeddings(split, always_calc_embeddings=always_calc_embeddings)
    elif model_id == 'pt5':
        from src.pretrained_embeddings.calc_pt5_embeddings import calc_embeddings
        Xs_positive, Xs_negative = calc_embeddings(model_id, split, always_calc_embeddings=always_calc_embeddings)
    else:  # esm
        from src.pretrained_embeddings.calc_esm_embeddings import calc_embeddings
        Xs_positive, Xs_negative = calc_embeddings(model_id, split, always_calc_embeddings=always_calc_embeddings)

    Xs = np.concatenate([Xs_positive, Xs_negative])
    Ys = [1] * Xs_positive.shape[0] + [0] * Xs_negative.shape[0]

    # Shuffle
    combined = list(zip(Xs, Ys))
    random.shuffle(combined)
    shuffled_Xs, shuffled_Ys = zip(*combined)
    shuffled_Xs = np.array(shuffled_Xs)

    logger.info(f"Loaded {split} data: Xs_{split}.shape = {Xs.shape}, Ys_{split}.shape = {len(Ys)}")

    return shuffled_Xs, shuffled_Ys
