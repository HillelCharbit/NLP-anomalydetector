
<div align="center">

# "Anomaly Forecast: Innovative Techniques for Future Insights and Normalization"

</div>

## ✨ Introduction

This project builds upon ["Chronos: Learning the Language of Time Series"](https://arxiv.org/abs/2403.07815) which uses transformer-based language models to forecast time series data by tokenizing values and training on diverse datasets.

 We extend Chronos in two significant ways. First, we apply the pre-trained models to anomaly detection, leveraging their forecasting capabilities to identify deviations in time series data, crucial for sectors like finance and cybersecurity. Second, we introduce two novel normalization techniques aimed at improving model performance and stability, addressing issues like token and attention shifts. Our experiments compare these extensions against established baselines across various datasets. Preliminary results show potential in enhancing anomaly detection and suggest further refinement of Chronos-based models for both forecasting and anomaly detection tasks.

For details on the training data and procedures, and experimental results, please refer to the project report [Anomaly Forecast: Innovative Techniques for Future Insights and Normalizations]().

(https://arxiv.org/abs/2403.07815).

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

# Clone the repository
git clone https://github.com/HillelCharbit/NLP-anomalydetector.git


### Anomaly Detection
### Normalization Techniques

## :floppy_disk: Datasets

Datasets used in the Chronos paper for pretraining and evaluation that we used in this project are available through the HuggingFace repos: [`autogluon/chronos_datasets`](https://huggingface.co/datasets/autogluon/chronos_datasets) and [`autogluon/chronos_datasets_extra`](https://huggingface.co/datasets/autogluon/chronos_datasets_extra). Check out these repos for instructions on how to download and use the datasets.

## 🔥 Coverage

- ["Chronos: Learning the Language of Time Series"](https://arxiv.org/abs/2403.07815) 
- [Abnormality Forecasting: Time Series Anomaly Prediction via Future Context Modeling](https://arxiv.org/pdf/2410.12206v1)



## 📝 Citation

If you find this project useful for your research, please consider citing the associated contributers titled in this project.

## 📃 License

This project is licensed under the MIT licence. . For more details, please refer to the License file.
