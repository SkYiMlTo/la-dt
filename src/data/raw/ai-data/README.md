# AI Dataset

This directory contains the AI (Actual Implantation) dataset files used for training and evaluation.

## Expected Files

- `scaled_PV_data.csv` - Scaled photovoltaic data
- `scaled_load_data.csv` - Scaled load data
- `Q_load_data.csv` - Reactive power load data
- `Vpmu_data_489.csv` - Voltage PMU data for 489 buses
- `IAngpmu_indx_489.csv` - Current angle PMU indices
- `IMagpmu_indx_489.csv` - Current magnitude PMU indices
- `usedbusarray_1000plus_revised.xlsx` - Bus configuration
- `DNN_J1_489PMU_05TVE.h5` - Pre-trained DNN model
- `x_train12/` - Training features
- `y_train12` - Training labels
- `x_test_12.csv` - Test features

## Usage

These files are used in experiments for testing the anomaly detection model on the AI dataset.
