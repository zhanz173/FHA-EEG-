
import torch
import numpy as np
from dataloader import EEGDatasetWithLabel
from ensemble_trainer import DeepEnsembleTrainer, TrainConfig
from model import GRU_Classifier, NeurologistCorrectionConfig

def build_dataset(dataset_csv_path, experiment_name="example"):
    all_data = pd.read_csv(dataset_csv_path)
    unique_patients = all_data['Hashed_PatientURN'].unique()
    # 70% train, 30% evaluation split, skip test for now 
    train_size = int(0.7 * unique_patients.shape[0])
    val_size = unique_patients.shape[0] - train_size
    # patients in training set needs to be disjoint from those in validation and test sets
    patient_ids_trainset = np.random.choice(unique_patients, size=train_size, replace=False)
    patient_ids_valset = np.setdiff1d(unique_patients, patient_ids_trainset)

    train_set = all_data[all_data['Hashed_PatientURN'].isin(patient_ids_trainset)]
    val_set = all_data[all_data['Hashed_PatientURN'].isin(patient_ids_valset)]
    print(f"Train set size: {train_set.shape[0]}")
    print(f"Validation set size: {val_set.shape[0]}")

    # save train and val sets to separate CSV files
    os.makedirs(f"{experiment_name}", exist_ok=True)
    train_set.to_csv(rf"{experiment_name}\train_data.csv", index=False)
    val_set.to_csv(rf"{experiment_name}\eval_data.csv", index=False)

def validate_dataset_splits(experiment_name="checkpoints_run"):
    import pandas as pd
    train = pd.read_csv(f"{experiment_name}/train_data.csv")
    eval = pd.read_csv(f"{experiment_name}/eval_data.csv")
    train_SHA = train['Hashed_PatientURN'].unique()
    val_SHA = eval['Hashed_PatientURN'].unique()

    assert len(set(train_SHA).intersection(set(val_SHA))) == 0, "Data leakage between train and validation!"
    print("No data leakage between splits.")


def model_gen(args:dict):
    hidden_size = args.get("hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    num_classes = args.get("num_classes", 5)
    neurologist_correction_config = args.get("neurologist_correction_config", None)
    pooling_output_size = args.get("pooling_output_size", 64)
    model = GRU_Classifier(hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes, neurologist_correction_config=neurologist_correction_config, pooling_output_size=pooling_output_size)
    return model

def train(trainer: DeepEnsembleTrainer, train_ds, val_ds, config: TrainConfig, ensemble_size=3):
    result = trainer.fit_ensemble(train_ds, val_ds, ensemble_size=ensemble_size, cfg=config, use_bootstrap_sampler=False, verbose=True)
    return result



if __name__ == "__main__":
    import os
    import pandas as pd

    FHA_EEG_FEATURES_ROOT = r"H:\EEG_features\EEG_features_labram_welch_4s"
    FHA_EEG_METADATA_ROOT = r"H:\EEG\FHA\label_matched_metadata.csv"
    EXPERIMENT_NAME = r"example"
    build_dataset(dataset_csv_path=FHA_EEG_METADATA_ROOT, experiment_name=EXPERIMENT_NAME)
    validate_dataset_splits(experiment_name=EXPERIMENT_NAME)  # check for data leakage

    CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
    test_training_config = TrainConfig(
        epochs=15,
        use_positive_weight=False, # default use focal loss, no pos weight needed
        use_sample_weight=False,
        use_good_labels_only=False,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-5,
        amp=False,
        early_stop_patience=10,
        num_workers=4,
        bootstrapping_targets=0.9,    )
    
    ## log training config
    with open(f"{EXPERIMENT_NAME}/training_config.txt", "w") as f:
        f.write(str(test_training_config))

    print("Initializing training datasets...")
    train_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=f"{EXPERIMENT_NAME}/train_data.csv", return_ids=True,return_ordinal=False, return_neurologist_ids=True)
    print("Initializing validation datasets...")
    val_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=f"{EXPERIMENT_NAME}/eval_data.csv", return_ids=True, return_ordinal=False, return_neurologist_ids=True)
    
    trainer = DeepEnsembleTrainer(
        model_fn=model_gen,
        num_classes=5,
        model_kwargs={"hidden_size": 64, "num_layers": 3, "num_classes": 5, "num_pool_heads": 3, "pooling_output_size": 64},
        device="cuda" ,
        checkpoint_dir=f"{EXPERIMENT_NAME}/",
    )

    results = train(trainer, train_ds, val_ds, test_training_config)

    unc_test, metrics = trainer.ensemble_uncertainty(val_ds, checkpoint_dir=f"{EXPERIMENT_NAME}/")
    unc_test.to_csv(rf"{EXPERIMENT_NAME}/uncertainty_evalsamples.csv", index=False)