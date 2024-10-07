# hssm_mice_choice_history_ddm.py - a reproduction of models from
# Urai AE, de Gee JW, Tsetsos K, Donner TH (2019) 
# Choice history biases subsequent evidence accumulation. eLife
# using The International Brain Laboratory data
#
# Record of Revisions
#
# Date            Programmers                                Descriptions of Change
# ====         ================                              ======================
# 2024/10/06      Kianté Fernandez<kiantefernan@gmail.com>   updated workflow from hssm_reproduce for mice

import os
import shutil
import numpy as np
import pandas as pd
from utils_hssm_modelspec import make_model 
from utils_hssm import run_model

def load_and_preprocess_mouse_data(file_path):
    mouse_data = pd.read_csv(file_path)
    mouse_data['response'] = mouse_data['response'].replace({0: -1, 1: 1})
    mouse_data['rt'] = mouse_data['rt'].round(6)
    mouse_data['participant_id'] = pd.factorize(mouse_data['subj_idx'])[0] + 1
    mouse_data['repeat'] = np.where(mouse_data.response == mouse_data.prevresp, 1, 0)
    
    # Clean data
    mouse_data = mouse_data.dropna(subset=['rt', 'response', 'prevresp'])
    mouse_data = mouse_data[mouse_data['rt'] >= 0]
    
    return mouse_data

def print_dataset_stats(dataset):
    print(f"\nDataset summary:")
    print(f"Number of trials: {len(dataset)}")
    print(f"Number of unique subjects: {dataset['subj_idx'].nunique()}")
    print(f"RT range: {dataset['rt'].min():.3f} to {dataset['rt'].max():.3f} seconds")

    print("\nUnique subject IDs (subj_idx) in final dataset:")
    print(dataset['subj_idx'].unique())

    print("\nNumber of trials per subject in final dataset:")
    print(dataset['subj_idx'].value_counts())

    print("\nMapping between subj_idx and participant_id:")
    print(dataset[['subj_idx', 'participant_id']].drop_duplicates().sort_values('participant_id'))

def move_models_to_folder(src_dir, dest_dir):
    for filename in os.listdir(src_dir):
        if filename.endswith("_model.nc"):
            src_file = os.path.join(src_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            if os.path.exists(dest_file):
                print(f"Model {filename} already exists in destination. Skipping.")
            else:
                shutil.move(src_file, dest_file)
                print(f"Moved {filename} to {dest_dir}")

def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_file_path = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'data')
    results_dir = os.path.join(script_dir, '..', '..', '2023_choicehistory_HSSM', 'results')
    models_dir = os.path.join(results_dir, 'models')

    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Move any existing models from src to models folder
#    move_models_to_folder(script_dir, models_dir)

    # Load and preprocess mouse data
    mouse_data_path = os.path.join(data_file_path, 'ibl_trainingChoiceWorld_clean_20241003.csv')
    mouse_data = load_and_preprocess_mouse_data(mouse_data_path)
    columns_to_use = ['subj_idx', 'participant_id', 'rt', 'response', 'signed_contrast', 'prevresp', 'eid']
    dataset = mouse_data[columns_to_use]
    
    # Print dataset stats
    print_dataset_stats(dataset)
    
    # Define models to estimate
    model_names = ["ddm_nohist", "ddm_prevresp_v", "ddm_prevresp_z", "ddm_prevresp_zv"]
    
    # Parameters for sampling
    sampling_params = {"chains": 4, "cores": 4, "draws": 2000, "tune": 2000}
    
    # Run models
    for name in model_names:
        model_file = os.path.join(models_dir, f"{name}_model.nc")
        if os.path.exists(model_file):
            print(f"Model {name} already exists. Skipping.")
        else:
            print(f"Running model: {name}")
            model = run_model(dataset, name, script_dir, **sampling_params)
            # test_make_model = make_model(dataset, name)
            print(f"Model {name} completed and saved.")

if __name__ == "__main__":
    main()
    
    