"""
check_split.py

- To check correct train/val/test splits produced by preprocessing/data_preprocessing.py
- To verify that a patient by patient ID appears in no more than one of train/val/test splits

cmd line: 
- python3 preprocessing/verify_split.py
"""

import os

def get_patient_id(filename: str) -> str:
    return filename.split("_")[0]

def get_patients(folder):
    patients = set()
    total_files = 0

    for filename in os.listdir(folder):
        if not filename.endswith(".pkl"):
            continue
        total_files += 1
    
        patient_id = get_patient_id(filename)
        patients.add(patient_id)    
        
    return patients, total_files


def main():
    root = "./data/processed"

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    test_dir = os.path.join(root, "test")

    print("----- Dataset split -----")

    train_patient_ids, train_files = get_patients(train_dir)
    val_patient_ids, val_files = get_patients(val_dir)
    test_patient_ids, test_files = get_patients(test_dir)
    total_patients =  train_patient_ids | val_patient_ids | test_patient_ids

    print(f"Total patients: {len(total_patients)}")

    # patient and chunks count per set
    print(
        f"Train: {len(train_patient_ids):} patients "
        f"({len(train_patient_ids)/len(total_patients) * 100:.2f}%), "
        f"{train_files:} chunks"
    )

    print(
        f"Val: {len(val_patient_ids):} patients "
        f"({len(val_patient_ids)/len(total_patients) * 100:.2f}%), "
        f"{val_files:} chunks"
    )

    print(
        f"Test: {len(test_patient_ids):} patients "
        f"({len(test_patient_ids)/len(total_patients) * 100:.2f}%), "
        f"{test_files:} chunks"
    )

    # check patient level data leakage
    train_val = train_patient_ids & val_patient_ids
    train_test = train_patient_ids & test_patient_ids
    val_test = val_patient_ids & test_patient_ids

    print ("\n ----- Patient overlap check ------")

    print(f"Train & val: {len(train_val)} patients")
    print(f"Train & test: {len(train_test)} patients")
    print(f"Val & test: {len(val_test)} patients")

    if train_val:
        print(f"\n FAIL. Patient leakage between train and val: {sorted(train_val)}")
    
    if train_test:
        print(f"\n FAIL. Patient leakage between train and test: {sorted(train_test)}")
    
    if val_test:
        print(f"\n FAIL. Patient leakage between train and test: {sorted(val_test)}")

    if not train_val and not train_test and not val_test:
        print("\nPASS. Not patient overlap between train/val/test")
    else:
        print("\nFAIL. Patient overlap detected.")
    
if __name__ == "__main__":
    main()