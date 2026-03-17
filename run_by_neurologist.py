import os
import torch
import numpy as np
import pandas as pd
from src.dataloader import EEGDatasetWithLabel
from src.ensemble_trainer import DeepEnsembleTrainer, TrainConfig

def build_dataset(dataset_csv_path, directory="example"):
    all_data = pd.read_csv(dataset_csv_path)
    # select top 5 doctors
    top_5_doctors = ["Sophia", "Eleni", "Maria", "Zoe", "Athina"]
    EEG_top_5_doctors = all_data[all_data['Physician'].isin(top_5_doctors)]
    EEG_rest_doctors = all_data[~all_data['Physician'].isin(top_5_doctors)]

    # create a baseline, train on data regardless of doctor
    train_patients_all_doctors = []

    for doctor in top_5_doctors:
        doctor_data = EEG_top_5_doctors[EEG_top_5_doctors['Physician'] == doctor]
        unique_patients = doctor_data['Hashed_PatientURN'].unique()
        train_size = int(0.5 * unique_patients.shape[0])
        patient_ids_trainset = np.random.choice(unique_patients, size=train_size, replace=False)
        train_patients_all_doctors.extend(patient_ids_trainset) # accumulate train patients across doctors for training the baseline model
        doctor_eval_data = doctor_data[~doctor_data['Hashed_PatientURN'].isin(patient_ids_trainset)]
        doctor_train_data = doctor_data[doctor_data['Hashed_PatientURN'].isin(patient_ids_trainset)] # train on single doctor, evaluate on rest of the patients labeled by that doctor
        eval_data = pd.concat([EEG_rest_doctors, doctor_eval_data], axis=0)

        os.makedirs(f"{directory}/{doctor}", exist_ok=True)
        doctor_train_data.to_csv(rf"{directory}/{doctor}/train.csv", index=False)
        eval_data.to_csv(rf"{directory}/{doctor}/eval.csv", index=False)

    # baseline model trained on all doctors, evaluated on all patients not in the training set regardless of doctor
    train_patients_all_doctors = np.array(train_patients_all_doctors)
    baseline_train_data = all_data[all_data['Hashed_PatientURN'].isin(train_patients_all_doctors)]
    baseline_eval_data = all_data[~all_data['Hashed_PatientURN'].isin(train_patients_all_doctors)]
    os.makedirs(f"{directory}/Baseline", exist_ok=True)
    baseline_train_data.to_csv(rf"{directory}/Baseline/train.csv", index=False)
    baseline_eval_data.to_csv(rf"{directory}/Baseline/eval.csv", index=False)

def model_gen(args:dict):
    from src.model import GRU_Classifier
    encoder_hidden_size = args.get("encoder_hidden_size", 64)
    RNN_hidden_size = args.get("RNN_hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    num_classes = args.get("num_classes", 5)
    neurologist_correction_config = args.get("neurologist_correction_config", None)
    pooling_output_size = args.get("pooling_output_size", 64)
    pretrained_encoder_path = args.get("pretrained_encoder_path", None)
    model = GRU_Classifier(encoder_hidden_size=encoder_hidden_size, RNN_hidden_size=RNN_hidden_size, num_layers=num_layers, num_classes=num_classes, neurologist_correction_config=neurologist_correction_config, pooling_output_size=pooling_output_size)
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

def run(train_config, experiment_name="example", dataset_csv_path=r"H:\EEG\FHA\label_matched_metadata.csv"):
    # first build dataset splits 
    os.makedirs(f"{experiment_name}", exist_ok=True)

    ## log training config
    with open(f"{experiment_name}/training_config.txt", "w") as f:
        f.write(str(train_config))

    build_dataset(dataset_csv_path=dataset_csv_path, directory=experiment_name)

    #  we should have 6 sets of train and eval csv files, one for each of the top 5 doctors and one for the baseline model trained on all doctors
    for doctor in [ "Baseline", "Sophia", "Eleni", "Maria", "Zoe", "Athina"]: #hack: train a baseline model first. The sub-doctor models will be initialized with the encoder weights from the baseline model to stabilize training given the smaller dataset sizes
        print(f"Initializing training datasets for {doctor}...")
        train_ds = EEGDatasetWithLabel(root=r"H:\EEG_features\EEG_features_labram_welch_4s", metadata=f"{experiment_name}/{doctor}/train.csv", return_ids=True,return_ordinal=False, return_neurologist_ids=True)
        print(f"Initializing validation datasets for {doctor}...")
        val_ds = EEGDatasetWithLabel(root=r"H:\EEG_features\EEG_features_labram_welch_4s", metadata=f"{experiment_name}/{doctor}/eval.csv", return_ids=True, return_ordinal=False, return_neurologist_ids=True)
        
        trainer = DeepEnsembleTrainer(
            model_fn=model_gen,
            num_classes=5,
            model_kwargs={"encoder_hidden_size": 64, "RNN_hidden_size": 64, "num_layers": 2, "num_classes": 5, "num_pool_heads": 3, "pooling_output_size": 64, "pretrained_encoder_path": f"{experiment_name}/Baseline/member00.pt" if doctor != "Baseline" else None}, # for the doctor-specific models, initialize the encoder with weights from the baseline model trained on all doctors to stabilize training
            device="cuda" ,
            checkpoint_dir=f"{experiment_name}/{doctor}/",
        )
        results = train(trainer, train_ds, val_ds, train_config, ensemble_size=1)
        # after training, evaluate uncertainty on validation set and save results
        unc_test, metrics = evaluate_uncertainty(trainer, val_ds, checkpoint_dir=f"{experiment_name}/{doctor}/")
        unc_test.to_csv(rf"{experiment_name}/{doctor}/uncertainty_evalsamples.csv", index=False)

if __name__ == "__main__":
    import os
    import sys
    import pandas as pd
    sys.path.append(os.path.abspath("../src"))  # add src/ to path so we can import from it
    FHA_EEG_channels_ORDER =['C3','C4','Cz','F3','F4','F7','F8','Fz','Fp1','Fp2','Fpz','O1','O2','P3','P4','Pz','T3','T4','T5','T6']

    FHA_EEG_FEATURES_ROOT = r"H:\EEG_features\EEG_features_labram_welch_4s"
    FHA_EEG_METADATA_ROOT = r"E:\project\FHA-EEG-\data\full_matched.csv"
    EXPERIMENT_NAME = r"Nurologist_Correction_Experiment_4"

    CURRENT_FILEPATH = os.path.dirname(os.path.abspath(__file__))
    test_training_config = TrainConfig(
        epochs=12,
        use_positive_weight=False, # default use focal loss, no pos weight needed
        use_sample_weight=False,
        use_good_labels_only=False,
        batch_size=64,
        lr=1e-3,
        weight_decay=1e-5,
        amp=False,
        early_stop_patience=10,
        num_workers=4,
        bootstrapping_targets=0.9
    )
    run(test_training_config, experiment_name=EXPERIMENT_NAME, dataset_csv_path=FHA_EEG_METADATA_ROOT)
