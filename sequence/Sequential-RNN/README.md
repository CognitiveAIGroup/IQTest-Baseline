# IQtest-sequence
- This is a baseline for solving sequence problem in IQ tests. The implementation uses RNN model.
## Experiment Environment
- The code is tested with Python 3.6 and Pytorch 0.4.0
## Code Usage
#### Preprocessing Data
To preprocess data, run
`python data_utils.py`
- This file take the question file that contains stem, options, category and id, and the answer file that contains answer and hint for a question, then combine the stem and answer together for training. The outputed two files are train set and valid set.

#### Train Model
You can run the model with specified hyper-parameters
an example command is 
`python main.py --data_dir data --output_dir res --lr 1e-3 --hidden_size 8 --niters 1e5 --norm 1 --save 1`

- The file `main.py` trains an RNN model to solve sequence problems in IQ-test, and display losses on train set and valid set, and accuracy on valid set to evaluate performance. The visulized result and trained model are saved. Sequences are splited into sub-sequences with length 3 to train the model.

- The file `modeling.py` contains two data classes, which uses MinMaxScaler or divide by 10**MaxNumberLength to normalize data. This file also contains an RNN model called SeqRNN to solve seqence problem in IQ-test

- The file `eval.py` is used to generate answers for accuracy testing.
