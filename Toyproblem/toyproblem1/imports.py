# ------------------------------------------------------------------------- #
#                                   Imports                                 #
# ------------------------------------------------------------------------- #
import os
import pandas as pd

# ------------------------------------------------------------------------- #
#                                  Functions                                #
# ------------------------------------------------------------------------- #

def posture(df_dir, df_name='df_raw'):
    df_path = os.path.join(df_dir, f"{df_name}.csv")
    try:
        df = pd.read_csv(df_path,
                        parse_dates=['Timestamp'],
                        dayfirst=True,
                        date_format='%Y-%m-%d %H:%M:%S.%f',
                        dtype = {'Subject': str, 'Position': str, 'Breed': str})  # Usar date_format
    except:
        df = pd.read_csv(df_path,
                        parse_dates=['Timestamp'],
                        dayfirst=True,
                        date_format='%Y-%m-%d %H:%M:%S.%f')  # Usar date_format
    return df
