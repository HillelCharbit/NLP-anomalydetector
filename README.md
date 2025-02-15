
<div align="center">

# Anomaly Forecast: Innovative Techniques for Future Insights and Normalization

</div>

## ✨ Introduction

This project builds upon ["Chronos: Learning the Language of Time Series"](https://arxiv.org/abs/2403.07815) which uses transformer-based language models to forecast time series data by tokenizing values and training on diverse datasets.

 We extend Chronos in two significant ways. First, we apply the pre-trained models to anomaly detection, leveraging their forecasting capabilities to identify deviations in time series data, crucial for sectors like finance and cybersecurity. Second, we introduce two novel normalization techniques aimed at improving model performance and stability, addressing issues like token and attention shifts. Our experiments compare these extensions against established baselines across various datasets. Preliminary results show potential in enhancing anomaly detection and suggest further refinement of Chronos-based models for both forecasting and anomaly detection tasks.

For details on the training data procedures and experimental results, please refer to the project report [Anomaly Forecast: Innovative Techniques for Future Insights and Normalizations]().


<p align="center">

 <img src="https://github.com/HillelCharbit/NLP-anomalydetector/blob/main/anomaly_detection/Pipeline_flowchart.png?raw=true" alt="Alt Text" style="width:auto; height:30%;">
  <br />
  <span>
    Fig. 1: Anomaly Detection Pipeline Overview. The process begins with an input-labeled dataset, which is split into training and test sets. The training data follows two paths: raw and pre-processed. The Chronos Forecaster is applied to both paths. For the pre-processed data, it directly forecasts anomalies. For the unprocessed data, it first performs a time series forecast, followed by state-of-the-art (SOTA) anomaly detection. Both approaches lead to prediction evaluation, which is then compared against the test data for validation.
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

Use the suggested datasets or any binary labeled anomaly time series CSV file. 

### Normalization Techniques

The normalization_adaptation folder contains all scripts and notebooks to reproduce the code of the normalization technique modification of Chronos.

#### Training

For training, we refer to the  `README.md` found in  `chronos-forecasting/scripts`. Also many additional details can be found in the [GitHub repo of Chronos](https://github.com/amazon-science/chronos-forecasting/tree/main).

Additionally, modify the config files in the same folder for different hyperparameters and change the tokenizer class for training on the specific normalization technique. 

#### Evaluation

To run the evaluation, use the notebook  `eval_pipeline.ipynb`. In the pipeline, specify what you want to use (either locally or pre-trained, see the **Training** section on how to use locally trained models). Also, specify the device type (CPU or GPU) and the Tokenizer Class. All these parameters can be specified in the **Global Params** section. Lastly, just run all cells and the evaluation should start. You can see the progress as the output of the last cell. The results are written into a CSV file where each row represents a dataset and its computed metrics.

The following pre-trained models are available:

| Model                                                                  | Parameters | Based on                                                               |
| ---------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| [**chronos-t5-tiny**](https://huggingface.co/amazon/chronos-t5-tiny)   | 8M         | [t5-efficient-tiny](https://huggingface.co/google/t5-efficient-tiny)   |
| [**chronos-t5-mini**](https://huggingface.co/amazon/chronos-t5-mini)   | 20M        | [t5-efficient-mini](https://huggingface.co/google/t5-efficient-mini)   |
| [**chronos-t5-small**](https://huggingface.co/amazon/chronos-t5-small) | 46M        | [t5-efficient-small](https://huggingface.co/google/t5-efficient-small) |
| [**chronos-t5-base**](https://huggingface.co/amazon/chronos-t5-base)   | 200M       | [t5-efficient-base](https://huggingface.co/google/t5-efficient-base)   |
| [**chronos-t5-large**](https://huggingface.co/amazon/chronos-t5-large) | 710M       | [t5-efficient-large](https://huggingface.co/google/t5-efficient-large) |

#### Modified Code

Next to the evaluation code, the code of the new normalization techniques can be found in ```src/chronos/chronos.py``` inherited from ```ChronosTokenizer```.

## 📃 License

This project is licensed under the MIT license. For more details, please refer to the license file.
