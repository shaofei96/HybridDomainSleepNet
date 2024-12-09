# HybridDomainSleepNet

This study implements the proposed HybridDomainSleepNet model validation with the public Montreal Archive of Sleep Studies (MASS)-SS3 and DREAMS Patients (DRM-Pat) datasets. Simultaneous validation was also performed using a homemade sleep dataset, the BP-SleepX.

Of these, the MASS-SS3 dataset can be accessed through the link http://ceams-carsm.ca/mass/ , and the DRM-Pat dataset can be accessed through the link https://zenodo.org/records/2650142. 

In addition, the homemade BP-SleepX dataset can be used by sending an email request for authorization to the corresponding author.

## Descriptions of the provided files:

In the following, brief descriptions of all provided files are given:
- `sleep_process.py` :    a function to capture the timing fragments.
- `DE.py` : a function to calculate the Differential Entropy (DE).
- ` data_load.py` : a function to load data.
- `model.py` : implements the proposed HybridDomainSleepNet described in the paper.
- `Index_calculation.py` : a function to calculate  the model prediction results.


The provided codes have been written in the following settings:  根据实际的修改一下

- matplotlib == 3.7.0
- mne ==1.4.2
- pandas == 2.2.3
- torch == 1.12.0+cu116
- scikit-learn == 1.2.2
- numpy == 1.25.2

