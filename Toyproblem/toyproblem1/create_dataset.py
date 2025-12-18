# ------------------------------------------------------------------------- #
#                           Importing Global Modules                        #
# ------------------------------------------------------------------------- #
import os, sys
from datetime import timedelta
import tqdm
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------- #
#                           Importing Local Modules                         #
# ------------------------------------------------------------------------- #

dir_current = os.path.dirname(os.path.realpath(__file__))
dir_parent = os.path.dirname(dir_current)
dir_base = os.path.dirname(dir_parent)
dir_modules = os.path.join(dir_base, 'src', '__modules__')

dir_raw = os.path.join(dir_base, 'data', 'raw')

# Set path variable
sys.path.append(dir_modules)

import imports, process


# ------------------------------------------------------------------------- #
#                           Setting Window Config                           #
# ------------------------------------------------------------------------- #

w_size = 100
w_overlap = .5
t_time = timedelta(seconds = .25)

# ------------------------------------------------------------------------- #
#                              Importing Chest Back Dataset                 #
# ------------------------------------------------------------------------- #

print("=== Starting Chest Back Dataset Processing ===")

# importing created raw dataset
back = imports.posture(dir_raw, 'chest_back_raw')

dir_old = os.path.join(dir_base, 'data', 'tsfel/chest_back')
dir_new = os.path.join(dir_base, 'data', 'final/chest_back')

test_back = ["66", "73"]
dev_back = ["58", "20", "45", "51", "65", "44", "28", "34", "61"]



# ------------------------------------------------------------------------- #
#                              Feature Extraction                           #
# ------------------------------------------------------------------------- #

process.features_tsfel(back, dir_old, w_size, w_overlap, t_time)

# import all
for df_name in tqdm.tqdm(os.listdir(dir_old), desc="Processing files"):

        df_old = imports.posture(dir_old, df_name[:-4])

        # select rows with the 5 position type for learning
        df_new = df_old[df_old.Position.isin(['standing', 'walking',
                  'sitting', 'lying down', 'body shake'])]

        # Save to the respective file
        if any(subject in df_new.Subject.unique() for subject in dev_back):
                df_new.to_csv(os.path.join(dir_new, 'df-all-dev.csv'), mode='a', header=not os.path.exists(os.path.join(dir_new, 'df-all-gr.csv')))

        elif any(subject in df_new.Subject.unique() for subject in test_back):
                df_new.to_csv(os.path.join(dir_new, 'df-all-test.csv'), mode='a', header=not os.path.exists(os.path.join(dir_new, 'df-all-test.csv')))

# Calculate proportions
dev_file = os.path.join(dir_new, 'df-all-dev.csv')
test_file = os.path.join(dir_new, 'df-all-test.csv')

if os.path.exists(dev_file):
    dev_data = pd.read_csv(dev_file)
    print(f"Proportion of dev dataset: {len(dev_data) / (len(dev_data) + len(pd.read_csv(test_file))):.2f}")

if os.path.exists(test_file):
    test_data = pd.read_csv(test_file)
    print(f"Proportion of test dataset: {len(test_data) / (len(dev_data) + len(test_data)):.2f}")

print("=== Finished Chest Back Dataset Processing ===")

# ------------------------------------------------------------------------- #
#                              Importing Marinara Dataset                   #
# ------------------------------------------------------------------------- #

print("=== Starting Marinara Dataset Processing ===")

# importing created raw dataset
marinara = imports.posture(dir_raw, 'marinara_raw')

dir_old = os.path.join(dir_base, 'data', 'tsfel/marinara')
dir_new = os.path.join(dir_base, 'data', 'final/marinara')

test_mr = ["11-23-IT"]
dev_mr = ["04-01-LD", "05-01-LD", "05-04-LZ", "10-15-LK", "10-03-LJ"]




# ------------------------------------------------------------------------- #
#                              Feature Extraction                           #
# ------------------------------------------------------------------------- #

marinara = marinara

process.features_tsfel(marinara, dir_old, w_size, w_overlap, t_time)

# import all
for df_name in tqdm.tqdm(os.listdir(dir_old), desc="Processing files"):

        df_old = imports.posture(dir_old, df_name[:-4])

        # select rows with the 5 position type for learning
        df_new = df_old[df_old.Position.isin(['standing', 'walking',
                  'sitting', 'lying down', 'body shake'])]

        # Save to the respective file
        if any(subject in df_new.Subject.unique() for subject in dev_mr):
                df_new.to_csv(os.path.join(dir_new, 'df-all-dev.csv'), mode='a', header=not os.path.exists(os.path.join(dir_new, 'df-all-gr.csv')))

        elif any(subject in df_new.Subject.unique() for subject in test_mr):
                df_new.to_csv(os.path.join(dir_new, 'df-all-test.csv'), mode='a', header=not os.path.exists(os.path.join(dir_new, 'df-all-test.csv')))

# Calculate proportions
dev_file = os.path.join(dir_new, 'df-all-dev.csv')
test_file = os.path.join(dir_new, 'df-all-test.csv')

if os.path.exists(dev_file):
    dev_data = pd.read_csv(dev_file)
    print(f"Proportion of dev dataset: {len(dev_data) / (len(dev_data) + len(pd.read_csv(test_file))):.2f}")

if os.path.exists(test_file):
    test_data = pd.read_csv(test_file)
    print(f"Proportion of test dataset: {len(test_data) / (len(dev_data) + len(test_data)):.2f}")

print("=== Finished Marinara Dataset Processing ===")
