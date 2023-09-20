import numpy as np
import pandas as pd
def Data_dispose(stress,strain,data_stress,data_strain):
    stress = np.array(stress)
    strain = np.array(strain)
    data_stress = np.array(data_stress)
    data_strain = np.array(data_strain)
    if stress.shape[0] != strain.shape[0]:
        print('stress : ',stress.shape)
        print('strain ; ',strain.shape)
        print('stress != strain')
        exit(1)
    if not (data_stress==data_strain).all() :
        print('data_stress != data_strain')
        exit(1)

    gen_data = np.hstack([stress,strain])
    data = data_stress
    if data.shape[0] != gen_data.shape[0]:
        print('stress_starin : ',gen_data.shape)
        print('data_all  : ',data.shape)
        print('data_all != stress   data_all != strain')
        exit(1)
    len = gen_data.shape[0]
    row = gen_data.shape[1]
    for index in range(len):
        for step in range(row):
            data[index, 64*128 + step] = gen_data[index, step]

    return data



