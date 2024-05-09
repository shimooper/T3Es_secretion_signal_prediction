import os
import sys
from timeit import default_timer as timer

import pandas as pd
from tensorflow import keras
from sklearn.metrics import average_precision_score, matthews_corrcoef

sys.path.append(os.path.join(os.path.dirname(__file__), 'protein_bert'))

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

from consts import OUTPUTS_DIR, BATCH_SIZE
from utils import read_train_data, read_test_data

EPOCHS = 1

# A local (non-global) binary output
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)


def test_on_test_data(model_generator, input_encoder, split):
    # Evaluating the performance on the test-set
    start_test_time = timer()

    test_sequences, test_labels = read_test_data(split=split)

    y_pred, results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, test_sequences,
                                                        test_labels, start_seq_len=512, start_batch_size=32)

    end_test_time = timer()
    elapsed_time = end_test_time - start_test_time

    y_pred_classes = (y_pred >= 0.5).astype(int)
    mcc = matthews_corrcoef(test_labels, y_pred_classes)
    auprc = average_precision_score(test_labels, y_pred)

    return mcc, auprc, elapsed_time


def main():
    # Loading the dataset
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()

    print(f'{len(train_sequences)} training set records, {len(validation_sequences)} validation set records')

    # Loading the pre-trained model and fine-tuning it on the loaded dataset
    pretrained_model_generator, input_encoder = load_pretrained_model(validate_downloading=False)

    # get_model_with_hidden_layers_as_outputs gives the model output access to the hidden layers (on top of the output)
    model_generator = FinetuningModelGenerator(pretrained_model_generator, OUTPUT_SPEC,
                                               pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs,
                                               dropout_rate=0.5)

    training_callbacks = [
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=1),
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
    ]

    finetune(model_generator, input_encoder, OUTPUT_SPEC, train_sequences, train_labels, validation_sequences,
             validation_labels,
             seq_len=512, batch_size=BATCH_SIZE, max_epochs_per_stage=EPOCHS, lr=1e-04, begin_with_frozen_pretrained_layers=True,
             lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
             callbacks=training_callbacks)

    test_mcc, test_auprc, test_elapsed_time = test_on_test_data(model_generator, input_encoder, 'test')
    xantomonas_mcc, xantomonas_auprc, xantomonas_elapsed_time = test_on_test_data(model_generator, input_encoder, 'xantomonas')

    # Save the results
    results_df = pd.DataFrame({
        'test_mcc': [test_mcc], 'test_auprc': [test_auprc], 'test_elapsed_time': [test_elapsed_time],
        'xantomonas_mcc': [xantomonas_mcc], 'xantomonas_auprc': [xantomonas_auprc], 'xantomonas_elapsed_time': [xantomonas_elapsed_time]
    })

    output_dir = os.path.join(OUTPUTS_DIR, 'protein_bert_finetune')
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'esm_finetune_results.csv'), index=False)


if __name__ == "__main__":
    main()
