# fashion-mnist
Project to deploy a machine learning model from a notebook based on fashion mnist dataset

## 1. Description
### 1.1 dataset
https://www.tensorflow.org/datasets/catalog/fashion_mnist
https://github.com/zalandoresearch/fashion-mnist

### 1.2 notebook
https://www.kaggle.com/viratkothari/image-classification-of-fashion-mnist-tensorflow

## 2. Solution
In this documentation I will refer to the service the serves the model as `predictor service`

| :warning: DISCLAIMERS|
| --- |
| I used MongoDB to store both input data and predictions because it is the best database (that I know) to store this type of data (n dimensional arrays) |
| Ideally the predictor service would be as simple as possible, which means it would not integrate with database, it could either send events or have middleware to do this integration. I saved predictions directly from predictor service to make implementation easier |
| Ideally the binary model would not be committed to github, there are better solutions to store it, for example S3. I saved on github to make implementation easier

### 2.1 Project structure
* data: directory where data is saved during training step
  * raw: raw data
  * processed: processed data
* models: where the tensorflow binary model is persisted
* notebooks: where the original notebook is located (for reference)
* scripts: all scripts used during training step
* src: predictor service code
  * api: codes related to api
  * pipeline: codes related to the prediction pipeline (pre-processing and training)
  * serializers: codes related to data serialization
* images: images used in README documentation   

### 2.2 Steps to deploy the model
