
import torch
import numpy as np
from src.dataloader import EEGDatasetWithLabel
from src.ensemble_trainer import DeepEnsembleTrainer, TrainConfig
from src.model import GRU_Classifier, NeurologistCorrectionConfig

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
    train_set.to_csv(rf"{experiment_name}\train.csv", index=False)
    val_set.to_csv(rf"{experiment_name}\eval.csv", index=False)

def validate_dataset_splits(experiment_name="checkpoints_run"):
    import pandas as pd
    train = pd.read_csv(f"{experiment_name}/train.csv")
    eval = pd.read_csv(f"{experiment_name}/eval.csv")
    train_SHA = train['Hashed_PatientURN'].unique()
    val_SHA = eval['Hashed_PatientURN'].unique()

    assert len(set(train_SHA).intersection(set(val_SHA))) == 0, f"Data leakage between train and validation!, number of overlapping patients: {len(set(train_SHA).intersection(set(val_SHA)))}"
    print("No data leakage between splits.")


def model_gen(args:dict):
    encoder_hidden_size = args.get("encoder_hidden_size", 64)
    RNN_hidden_size = args.get("RNN_hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    num_classes = args.get("num_classes", 5)
    neurologist_correction_config = args.get("neurologist_correction_config", None)
    pooling_output_size = args.get("pooling_output_size", 64)
    pretrained_encoder_path = args.get("pretrained_encoder_path", None)
    use_mean_pooling = args.get("use_mean_pooling", False)
    model = GRU_Classifier(encoder_hidden_size=encoder_hidden_size, RNN_hidden_size=RNN_hidden_size, num_layers=num_layers, num_classes=num_classes, neurologist_correction_config=neurologist_correction_config, pooling_output_size=pooling_output_size, use_mean_pooling=use_mean_pooling)
    if pretrained_encoder_path:
        print(f"Loading pretrained encoder from {pretrained_encoder_path}...")
        pretrained_state_dict = torch.load(pretrained_encoder_path, map_location='cpu')
        model.load_pretrained_encoder_weights(pretrained_state_dict["state_dict"])
    return model

def train(trainer: DeepEnsembleTrainer, train_ds, val_ds, config: TrainConfig, ensemble_size=2):
    result = trainer.fit_ensemble(train_ds, val_ds, ensemble_size=ensemble_size, cfg=config, use_bootstrap_sampler=False, verbose=True)
    return result


def evaluate_uncertainty(trainer: DeepEnsembleTrainer, val_ds, checkpoint_dir):
    unc_test, metrics = trainer.ensemble_uncertainty(val_ds, checkpoint_dir=checkpoint_dir)
    return unc_test, metrics

if __name__ == "__main__":
    import os
    import pandas as pd
    FHA_EEG_channels_ORDER =['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

    FHA_EEG_FEATURES_ROOT = r"H:\EEG_features\EEG_features_labram_welch_4s"
    FHA_EEG_METADATA_ROOT = r"E:\project\FHA-EEG-\data\full_matched.csv"
    EXPERIMENT_NAME = r"example"
    #build_dataset(dataset_csv_path=FHA_EEG_METADATA_ROOT, experiment_name=EXPERIMENT_NAME)

    CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
    nerologist_table_config = NeurologistCorrectionConfig (
        num_neurologists=5 + 1, # +1 for the "default" pseudo-rater representing the original model without neurologist correction, needs to be consistent with the neurologist IDs in the dataset
        rater_emb_dim=4,
        default_mean_rater=True)

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

    validate_dataset_splits(experiment_name=EXPERIMENT_NAME)  # check for data leakage
    print("Initializing training datasets...")
    train_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=f"{EXPERIMENT_NAME}/train.csv", return_ids=True,return_ordinal=False, return_neurologist_ids=True)
    print("Initializing validation datasets...")
    val_ds = EEGDatasetWithLabel(root=FHA_EEG_FEATURES_ROOT, metadata=f"{EXPERIMENT_NAME}/eval.csv", return_ids=True, return_ordinal=False, return_neurologist_ids=True)
    
    trainer = DeepEnsembleTrainer(
        model_fn=model_gen,
        num_classes=5,
        model_kwargs={"encoder_hidden_size": 128, "RNN_hidden_size": 64, "num_layers": 3, "num_classes": 5, "num_pool_heads": 4, "pooling_output_size": 64, "pretrained_encoder_path": None, "neurologist_correction_config": None, "use_mean_pooling": False},
        device="cuda" ,
        checkpoint_dir=f"{EXPERIMENT_NAME}/",
    )
    results = train(trainer, train_ds, val_ds, test_training_config, ensemble_size=1)

     # after training, evaluate uncertainty on validation set and save results
    unc_test, metrics = evaluate_uncertainty(trainer, val_ds, checkpoint_dir=f"{EXPERIMENT_NAME}/")
    unc_test.to_csv(rf"{EXPERIMENT_NAME}/uncertainty_evalsamples.csv", index=False)
