
# Comparative Analysis of Deep Learning Techniques on Electricity Time Series Data

## Abstract
Historically, time-series forecasting has been dominated by linear and ensemble methods since they are well-understood and highly effective on various problems when supported with feature engineering. Deep Learning techniques have enabled the learning of complex relations between sequential inputs and outputs with limited feature engineering. In this project, I have explored some of these techniques on the Electricity power dataset along with using some of the feature engineering techniques that are widely applied in time-series forecasting, such as one-hot encoding, lagging, and cyclical time features. 
Along with that, the study performs empirical analysis on the ensemble and the RNN models to understand the relative feature significance in predicting the electricity meter readings.

## Tech Stack

Python 3.9, Jupyter Notebook

## Project Report

The Project Report contains detailed analysis and explanation of 
all the literature and methodology. It is present in the ``report`` folder
titled ``Project_Report.pdf``


## Run Locally

Download the project or clone as below:

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd ds8013_ts_forecasting_project
```

Install dependencies

```bash
  pip install -r requirements.txt
```
Make sure data is downloaded and present ``data`` folder

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
Sample notebook with all results is present in ``reference_notebook`` folder. 
For plots that are missing, they are present in the ``plots`` folder.

``src/utils.py`` contains all the utility functions

## Feedback

If you have any feedback, please reach out to me at farzeemjiwani@gmail.com


## License

[MIT](https://choosealicense.com/licenses/mit/)
