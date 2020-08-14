# Solve Sequence Problem Using ANN 

## Description

This is a simple example of the baseline model for the IQtest, using the ANN algorithm, which is able to solve sequence predicting problem in the Sequence category. The model is only appliable for the sequence exactly consisting of numbers.

## Function

To find the next or a missing number, given a sequence of numbers.

## How to use

This model can realize prediction which just need file including sequences and accuracy calculation which need both sequence file and answer file.
Put the json file including sequences to be solved in your directory in certain form which will be explained later, and record the path as PATH1.
Also, if you have the answer of these incomplete sequences, you can put it in another json file in certain form which will be explained later, and record the path as PATH2.


### Formulate your sequence file

Please put the incomplete sequences in a .json file. The data should at least consist of "stem","options" and "id".
e.g.
>           "stem": "1,2,3,4,5,?"
>           "options": "3","4","5","6"
>           "id":0

### Formulate your answer file

Please put the answers in a .json file which should at least include "answers".
e.g.

>       "answers":[6]

### Input and output

Now you have recorded PATH1 (and PATH2), just take the path as parameter into the model as follow

```python
    Numlist=Numlist(PATH1,PATH2)
```
or
```python
    Numlist=Numlist(PATH1)
```
Then the predicted result is kept in **testPredict** and accuracy rate will be printed out.




