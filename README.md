# RandomLinearProjections

## Overview

This code implements the definition of a new loss function entitled Random Linear Projection Loss and provides a framework for running simulations on specified datasets for two primary tasks: Regression and Autoencoder comparing the performance of RLP to MSE and MSE+L2. it also includes two jupyter notebooks, one for a toy classification example (Moons Dataset), and the MNIST dataset. The user can customize several parameters including the dataset to use, task type, loss function, evaluation metric, and various training parameters. After running the simulations, the script will save the train and test losses to CSV files organized in respective directories.

## Requirements

- Python 3.x
- Libraries:
  - numpy
  - torch
  - sklearn


## Usage

To run the script, use the following command:

```
python3 main.py --Dataset "DatasetName" (optional arguments)
```

## Arguments

1. `--Dataset` (Required): Name of the dataset to be uploaded for the experiment.
2. `--Task` (Default: `Regression`): Specifies the task. It can be either `Regression` or `Autoencoder`.
3. `--LossFunction` (Default: `MSE`): Type of loss function to use. Options include: `MSE`, `MSEL2`, `RLP`, `MIXUP`, and `RLPMIX`.
4. `--TrainSize` (Default: `10000`): Size of the training dataset.
5. `--BatchSize` (Default: `100`): Size of each training batch.
6. `--NumBatches` (Default: `1000`): Number of RLP Loss Training Batches.
7. `--Epochs` (Default: `500`): Number of training epochs.
8. `--Iterations` (Default: `30`): Number of iterations to repeat the task.
9. `--Shift`: Dataset shift. Not required by default.
10. `--Noise`: Noise Scaling Factor. Not required by default.
11. `--EvalMetric` (Default: `MSE`): Evaluation metric. Can be either `MSE` or `RLP`.

## Output

After the simulation completes, the script saves the train and test losses in CSV files. The files are organized in directories based on the evaluation metric used (`MSE` or `RLP`). The naming convention for the files is:

```
Results/[EVAL_METRIC]/[TRAIN/TEST]_[DatasetName]_[task]_[loss_function]_TrainSize=[train_size]_BatchSize=[batch_size]_NumBatches=[num_batches]_Epochs=[epochs]_Iterations=[iterations]_Shift=[shift]_Noise=[noise].csv
```

## Errors

1. If the specified task is neither `Regression` nor `Autoencoder`, the script will output an error and exit.
2. If the evaluation metric specified is neither `MSE` nor `RLP`, the script will output an error and exit.

## Note

Before running the script, ensure that the required libraries are installed and that the path to the `RandomLinearProjections` directory is correctly set.

## Citation: 

Shyam Venkatasubramanian*, Ahmed Aloui*, Vahid Tarokh. "Random Linear Projections Loss for Hyperplane-Based Optimization in Regression Neural Networks".


