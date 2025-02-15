
<div align="center">

# Anomaly Forecast: Innovative Techniques for Future Insights and Normalization

</div>

## ✨ Introduction

This project builds upon ["Chronos: Learning the Language of Time Series"](https://arxiv.org/abs/2403.07815) which uses transformer-based language models to forecast time series data by tokenizing values and training on diverse datasets.

 We extend Chronos in two significant ways. First, we apply the pre-trained models to anomaly detection, leveraging their forecasting capabilities to identify deviations in time series data, crucial for sectors like finance and cybersecurity. Second, we introduce two novel normalization techniques aimed at improving model performance and stability, addressing issues like token and attention shifts. Our experiments compare these extensions against established baselines across various datasets. Preliminary results show potential in enhancing anomaly detection and suggest further refinement of Chronos-based models for both forecasting and anomaly detection tasks.

For details on the training data and procedures, and experimental results, please refer to the project report [Anomaly Forecast: Innovative Techniques for Future Insights and Normalizations]().


<p align="center">
  <img src="https://raw.githubusercontent.com/HillelCharbit/NLP-anomalydetector/anomaly_detection/Pipeline_flowchart.png" width="100%">
  <br />
  <span>
    Fig. 1: Anomaly Detection Pipeline Overview. The process begins with an input labeled dataset, which is split into training and test sets. The training data follows two paths: raw and pre-processed. The Chronos Forecaster is applied to both paths. For the pre-processed data, it directly forecasts anomalies. For the unprocessed data, it first performs a time series forecast, followed by state-of-the-art (SOTA) anomaly detection. Both approaches lead to prediction evaluation, which is then compared against the test data for validation.
  </span>
</p>

</div>

## 📈 Usage

If you're interested reproduce our experiments clone and install the environments from source:

Clone the repository:
```$ git clone https://github.com/HillelCharbit/NLP-anomalydetector.git```


Before running the evaluations, it is needed to install the required packages using conda. Therefore, we provide an environment yaml file in each of the folders.
Use the following command to create the environment:
```$ conda env create -f environment_name.yaml```

Activate the environment:
```$ conda activate environment_name```

### Anomaly Detection

TODO

### Normalization Techniques

The normalization_adaptation folder contains all scripts and notebooks to reporduce the code of the normalization technique modification of Chronos

#### Training

For training, we refer to the  `README.md` found in  `chronos-forecasting/scripts`. Also many additional details can be found in the [GitHub repo of Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main).

Additionally, modify the config files in the same folder for different hyperparameters and changing the tokenizer class for training on the specific normalization technique. 

#### Evaluation

To run the evaluation, use the notebook  `eval_pipeline.ipynb`. Therefore install all needed packages. Either you can create a new conda environment using the environment.yaml or you can install the package `chronos-forecasting` using your favorite package manager and then uninstalling it again. The uninstall keeps all packages but Chronos itself, as we are using the locally located code which contains the new normalization/tokenization classes. Finally, install the rest of the packages like gluonts, numpy etc. specified in the environment.yaml

In the pipeline, specify what you want to use (either locally or pretrained, see the **Training** section how to use locally trained models). Also specify the device type (CPU or GPU) and the Tokenizer Class. All these parameters can be specified in the **Global Params** section. Lasty, just run all cells and the evaluation should start. You can see the progress as output of the last cell. The results are written into a csv file where each row represents a dataset and its computed metrics.

#### Modified Code

Next to the evaluation code, the code of the new normalization techniques can be found in ```src/chronos/chronos.py``` inherited from ```ChronosTokenizer```.

## 📃 License

This project is licensed under the MIT license. For more details, please refer to the license file.
