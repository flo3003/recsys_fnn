# recsys_fnn
##Feedforward networks for recommender systems 

Make a new directory **results/** to store the results for each choice of optimization algorithm and its hyperparameters.

Store your splits in the following directory:

**my_data/dataset1/**

The first split will have the following names

u1.train, u1.test 

u1.train and u1.test must have **tab-separated** values with header: 

"msno" for the user id, "song_id" for the item id and "target" for the rating.

To run the *run_me.sh* script you need to specify which split . 

./run_me.sh 1

To run just one example e.g. 
Network: single_nn Learning rate: 0.01 Optimizer: SGD Activation function: selu Dropout: 0.05 Epochs: 100 Momentum: 0.2 Dataset: dataset1 Split: 1

python recsys_fnn.py single_nn 0.01 SGD selu 0.05 100 0.2 dataset1 1
