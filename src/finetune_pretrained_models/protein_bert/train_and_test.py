import os
import sys
from timeit import default_timer as timer

import pandas as pd
from tensorflow import keras
from sklearn.metrics import average_precision_score, matthews_corrcoef

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.utils.consts import FINETUNED_MODELS_OUTPUT_DIR, BATCH_SIZE, PROJECT_BASE_DIR, FINETUNE_NUMBER_OF_EPOCHS, \
    MODEL_ID_TO_MODEL_NAME, USE_LOCAL_MODELS, PROTEIN_BERT_MODEL_NAME, MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION
from src.utils.read_fasta_utils import read_train_data, read_test_data

sys.path.append(os.path.join(PROJECT_BASE_DIR, 'protein_bert'))

from proteinbert import OutputType, OutputSpec, FinetuningModelGenerator, load_pretrained_model, finetune, evaluate_by_len
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs


# A local (non-global) binary output
OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)


def evaluate_on_dataset(model_generator, input_encoder, sequences, labels):
    y_pred, results, confusion_matrix = evaluate_by_len(model_generator, input_encoder, OUTPUT_SPEC, sequences,
                                                        labels, start_seq_len=512, start_batch_size=32)

    y_pred_classes = (y_pred >= 0.5).astype(int)
    mcc = matthews_corrcoef(labels, y_pred_classes)
    auprc = average_precision_score(labels, y_pred)

    return mcc, auprc


def main():
    # Loading the dataset
    train_sequences, validation_sequences, train_labels, validation_labels = read_train_data()

    print(f'{len(train_sequences)} training set records, {len(validation_sequences)} validation set records')

    # Loading the pre-trained model and fine-tuning it on the loaded dataset

    model_id = 'protein_bert'
    pretrained_model_dir = MODEL_ID_TO_MODEL_NAME[model_id]
    if USE_LOCAL_MODELS:
        pretrained_model_generator, input_encoder = load_pretrained_model(
            local_model_dump_dir=pretrained_model_dir, local_model_dump_file_name=PROTEIN_BERT_MODEL_NAME)
    else:
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
             seq_len=512, batch_size=BATCH_SIZE, max_epochs_per_stage=FINETUNE_NUMBER_OF_EPOCHS, lr=1e-04, begin_with_frozen_pretrained_layers=True,
             lr_with_frozen_pretrained_layers=1e-02, n_final_epochs=1, final_seq_len=1024, final_lr=1e-05,
             callbacks=training_callbacks)

    train_mcc, train_auprc = evaluate_on_dataset(model_generator, input_encoder, train_sequences, train_labels)
    validation_mcc, validation_auprc = evaluate_on_dataset(model_generator, input_encoder, validation_sequences, validation_labels)

    start_test_time = timer()
    test_sequences, test_labels = read_test_data(split='test')
    test_mcc, test_auprc = evaluate_on_dataset(model_generator, input_encoder, test_sequences, test_labels)
    test_elapsed_time = timer() - start_test_time

    # Save the results
    results_df = pd.DataFrame({
        'train_mcc': [train_mcc], 'train_auprc': [train_auprc],
        'validation_mcc': [validation_mcc], 'validation_auprc': [validation_auprc],
        'test_mcc': [test_mcc], 'test_auprc': [test_auprc], 'test_elapsed_time': [test_elapsed_time],
        'model_id': [model_id],
        'training_mode': ['finetune'],
        'number_of_parameters (millions)': MODEL_ID_TO_PARAMETERS_COUNT_IN_MILLION[model_id]
    })

    output_dir = os.path.join(FINETUNED_MODELS_OUTPUT_DIR, model_id)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, f'best_model_results.csv'), index=False)


if __name__ == "__main__":
    main()
