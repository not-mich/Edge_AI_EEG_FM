import os
import pickle
from multiprocessing import Pool
import numpy as np
import mne

# 16 channel bipolar montage
BIPOLAR_PAIRS = [
    ("EEG FP1-REF", "EEG F7-REF"),
    ("EEG F7-REF", "EEG T3-REF"),
    ("EEG T3-REF", "EEG T5-REF"),
    ("EEG T5-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F8-REF"),
    ("EEG F8-REF", "EEG T4-REF"),
    ("EEG T4-REF", "EEG T6-REF"),
    ("EEG T6-REF", "EEG O2-REF"),
    ("EEG FP1-REF", "EEG F3-REF"),
    ("EEG F3-REF", "EEG C3-REF"),
    ("EEG C3-REF", "EEG P3-REF"),
    ("EEG P3-REF", "EEG O1-REF"),
    ("EEG FP2-REF", "EEG F4-REF"),
    ("EEG F4-REF", "EEG C4-REF"),
    ("EEG C4-REF", "EEG P4-REF"),
    ("EEG P4-REF", "EEG O2-REF"),
]

# process recordings belong to one patient
def split_and_dump(params):

    fetch_folder, patient_id, dump_folder, label = params

    for root_dir, _, files in os.walk(fetch_folder):

        for file in files:

            if not file.startswith(patient_id + "_"):
                continue

            if not file.endswith(".edf"):
                continue

            file_path = os.path.join(root_dir, file)

            try:
                # load .edf file
                raw = mne.io.read_raw_edf(
                    file_path,
                    preload=True,
                    verbose="ERROR"
                )

                # resampleto 200 Hz
                raw.resample(200)

                ch_names = raw.ch_names
                raw_data = raw.get_data()

                # verify all channels exist
                required_channels = set(sum(BIPOLAR_PAIRS, ()))

                if missing_channels:
                    raise ValueError(f"Missing channels: {missing_channels}")

                # create 16 channels
                channeled_data = np.zeros((len(BIPOLAR_PAIRS), raw_data.shape[1]), dtype=np.float32)

                for i, (ch1, ch2) in enumerate(BIPOLAR_PAIRS):
                    channeled_data[i] = (raw_data[ch_names.index(ch1)] - raw_data[ch_names.index(ch2)])

                # segment 10 sec windows at 200 Hz
                window_size = 2000

                total_slices = (
                    channeled_data.shape[1] // window_size
                )

                base_name = os.path.splitext(file)[0]

                for i in range(total_slices):

                    slice_data = channeled_data[:, i * window_size:(i + 1) * window_size]

                    dump_filename = (f"{base_name}_chunk_{i}.pkl")

                    dump_path = os.path.join( dump_folder, dump_filename)

                    with open(dump_path, "wb") as f:
                        pickle.dump(
                            { "X": slice_data, "y": label},
                            f
                        )
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

# create unique patient IDs from .edf file
def get_unique_patients(folder_path):
    patients = set()

    if not os.path.exists(folder_path):
        return []

    for root_dir, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".edf"):
                continue

            patient_id = file.split("_")[0]
            patients.add(patient_id)

    return list(patients)

def main():
    root_dir = "./data/tuh_eeg_abnormal/v3.0.0/edf"
    channel_std = "01_tcp_ar"

    train_abnormal_path = os.path.join( root_dir, "train", "abnormal", channel_std)
    train_normal_path = os.path.join(root_dir, "train", "normal", channel_std)
    eval_abnormal_path = os.path.join(root_dir, "eval", "abnormal", channel_std)
    eval_normal_path = os.path.join(root_dir, "eval", "normal", channel_std)

    # find patients
    train_a_patients = set(get_unique_patients(train_abnormal_path))
    train_n_patients = set(get_unique_patients(train_normal_path))
    eval_a_patients = set(get_unique_patients(eval_abnormal_path))
    eval_n_patients = set(get_unique_patients(eval_normal_path))

    # patient-lvel train/val split
    # combine train patients
    all_train_patients = list(train_a_patients | train_n_patients)

    rng = np.random.default_rng(42)
    rng.shuffle(all_train_patients)

    split_idx = int(len(all_train_patients) * 0.80)

    train_patients = set(all_train_patients[:split_idx])

    val_patients = set(all_train_patients[split_idx:])

    # verify overlap

    overlap = train_patients & val_patients

    if overlap:
        raise RuntimeError(f"Patient leakage detected before processing: {sorted(overlap)}")

    print(f"Total training patients: {len(all_train_patients)}")
    print(f"Training patients: {len(train_patients)}")
    print(f"Validation patients: {len(val_patients)}")
    print(f"Training abnormal patients: {len(train_a_patients)}")
    print(f"Training normal patients: {len(train_n_patients)}")

    # output folders
    processed_base = "./data/processed"
    train_dump = os.path.join(processed_base, "train")
    val_dump = os.path.join(processed_base, "val")
    test_dump = os.path.join(processed_base, "test")

    for folder in [train_dump, val_dump, test_dump]:
        os.makedirs(folder, exist_ok=True)

    parameters = []

    # train
    for patient_id in train_patients:
        if patient_id in train_a_patients:
            parameters.append((train_abnormal_path, patient_id, train_dump, 1))

        if patient_id in train_n_patients:
            parameters.append((train_normal_path, patient_id, train_dump, 0))

    # val
    for patient_id in val_patients:
        if patient_id in train_a_patients:
            parameters.append((train_abnormal_path, patient_id, val_dump, 1))

        if patient_id in train_n_patients:
            parameters.append((train_normal_path, patient_id, val_dump, 0))

    # test ("/eval" in TUAB dataset folder)
    for patient_id in eval_a_patients:
        parameters.append((eval_abnormal_path, patient_id, test_dump, 1))

    for patient_id in eval_n_patients:
        parameters.append((eval_normal_path, patient_id, test_dump, 0))

    print()
    print(f"Processing {len(parameters)}")

    with Pool(processes=os.cpu_count()) as pool:
        pool.map(split_and_dump, parameters)

    print(f"SUCCESS.Processed data saved to: {processed_base}")

if __name__ == "__main__":
    main()