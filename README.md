
# Comparative Analysis of Deep Learning Techniques on Electricity Time Series Data

## Abstract
Historically, time-series forecasting has been dominated by linear and ensemble methods since they are well-understood and highly effective on various problems when supported with feature engineering. Deep Learning techniques have enabled the learning of complex relations between sequential inputs and outputs with limited feature engineering. In this project, I have explored some of these techniques on the Electricity power dataset along with using some of the feature engineering techniques that are widely applied in time-series forecasting, such as one-hot encoding, lagging, and cyclical time features. 
Along with that, the study performs empirical analysis on the ensemble and the RNN models to understand the relative feature significance in predicting the electricity meter readings.

## Project Report

The Project Report contains detailed analysis and explanation of 
all the literature and methodology. It is present in the ``report`` folder
titled ``Project_Report.pdf``

## Documentation

``src`` folder contains all the codes.

``src/utils.py`` contains all the utility functions.

``data`` folder contains the electricity data

Reference notebook with all the results is present in ``reference_notebook`` folder titled ``Project_Notebook.ipynb`` which can be run directly on Google Colab or locally. 
For plots that are missing (as they are created in runtime using plotly), they are present in the ``plots`` folder.

``sample_rnn_models`` folder contains the RNN model binaries incase you want to skip the training of the models from scratch. 

``sample_results`` folder contains the predictions of all the models and their evaluation metrics.

## Tech Stack

Python 3.9, Jupyter Notebook

## Run Locally

Download the project or clone as below:

```bash
  git clone https://github.com/farzeemjiwani/ElectricityTSForecasting
```

Go to the project directory

```bash
  cd ElectricityTSForecasting
```

Install dependencies

```bash
  pip install -r requirements.txt
```
Make sure the data is downloaded and present in the ``data`` folder

Run no lag model

```bash
  cd src
  python no_lag_modeling.py 
```
Run lag model

```bash
  cd src
  python lag_modeling.py 
```

## Feedback

If you have any feedback, please reach out to me at farzeemjiwani@gmail.com

## License

[MIT](https://choosealicense.com/licenses/mit/)
