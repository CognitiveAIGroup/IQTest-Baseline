# IQ Test Baseline Model for Verbal Part
# Description

This is a simple example of the baseline model for the IQtest, which is able to solve analogy and classification questions in the Verbal category.

# How to use

## run_script.py

The run_script.py provides client_mode, server_mode, pack_model for the model
To customize the location of the extracted dataset, please specify  `DATA_ROOT`.

## entry_model.py

The model entry file that 

* defines get_eval_cls

>    ```python
>    def get_eval_cls(category: str) -> object:
>        pass
>    ```

* or defines get_model_object

>    ```python
>    def get_model_object(category: str) -> object:
>       pass
>    ```
>
>    get_model_object would override get_eval_cls
>    return a model class to each of the three categories, return `None` would ignore the corresponding competition

* provide global_pre_run (optional)

>    ```python
>    def global_pre_run():
>       pass
>    ```
>
>    Used for global setup, such as setup environment, load  train data so etc.

Model class will also need to provide

>    ```python
>    def solve(self, question):
>       return [question['id'], [1]]
>    ```
>
>    the variable 'question' will be a question from the question list that needs to be answerd by the model object, answers should be returned along with its question id

# Data

## group_data and train_data

please refer to README.md under source directory root.

The data required for this example is:
 ConceptNet Numberbatch 19.08(English-only):
This data contains semantic vectors from ConceptNet Numberbatch, by Luminoso Technologies, Inc. You may redistribute or modify the data under the terms of the CC-By-SA 4.0 license.

Please place the data under the the train_data directory.

For more information, visit [conceptnet-numberbatch](https://github.com/commonsense/conceptnet-numberbatch). Or [download](https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz).

## eval_data

The eval_data is the set of stardardized verbal questions we picked out from the dataset to work on.

 Feel free to run on it yourself, remember to  adjust the `DATA_ROOT` in run_script.py and adjust the "test_suites" in the config file.

