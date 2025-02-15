# Modified Chronos Fork for Normalization Technique Evaluation

This folder contains all scripts and notebooks to reporduce the code of the normalization technique modification of Chronos

## Training

For training, we refer to the  `README.md` found in  `chronos-forecasting/scripts`. Also many details in general can be found in the [GitHub repo of Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main).

Additionally, modify the config files in the same folder for different hyperparameters and changing the tokenizer class. 

## Evaluation

To run the evaluation, use the notebook  `eval_pipeline.ipynb`. Therefore install all needed packages. Either you can create a new conda environment using the environment.yaml or you can install the package `chronos-forecasting` using your favorite package manager and then uninstalling it again. The uninstall keeps all packages but Chronos itself, as we are using the locally located code which contains the new normalization/tokenization classes. Finally, install the rest of the packages like gluonts, numpy etc. specified in the environment.yaml

In the pipeline, specify what you want to use (either locally or pretrained, see the **Training** section how to use locally trained models). Also specify the device type (CPU or GPU) and the Tokenizer Class. All these parameters can be specified in the **Global Params** section. Lasty, just run all cells and the evaluation should start. You can see the progress as output of the last cell. The results are written into a csv file where each row represents a dataset and its computed metrics.