#DATASETS=("dataset1" "dataset2")

dataset="dataset1"
split=$1

NETWORKS=("single_nn" "single_nn_gm" "double_nn")

OPTIMIZERS=("SGD" "Adam" "RMSprop") #

SGD_LEARNING_RATES=("0.01" "0.02")
MOMENTUM_RATES=("0.0" "0.2" "0.5") #

REST_LEARNING_RATES=("0.0005" "0.0007" "0.001" "0.002" "0.003") #

ACTIVATIONS=("relu" "selu" "elu") #

SIMPLE_DROPOUTS=("0" "0.2" "0.5")
LPHA_DROPOUTS=("0" "0.05" "0.1") #


echo $filename

epochs=100

echo "network_type,result,optimizer,lr,momentum,dr,reg1,reg2,act_function,at_epoch" >> $filename

#for dataset in "${DATASETS[@]}"

#do

filename="results_"$dataset"_u"$split".csv"

for network in "${NETWORKS[@]}"

do

for optimizer in "${OPTIMIZERS[@]}" 

do 

if [ "$optimizer" == "SGD" ]

then

LEARNING_RATES=(${SGD_LEARNING_RATES[@]})

else

LEARNING_RATES=(${REST_LEARNING_RATES[@]})

fi

for lr in "${LEARNING_RATES[@]}"

do
			
for activation in "${ACTIVATIONS[@]}"

do

if [ "$activation" == "selu" ]

then

DROPOUTS=(${ALPHA_DROPOUTS[@]})

else

DROPOUTS=(${SIMPLE_DROPOUTS[@]})

fi

for dropout in "${DROPOUTS[@]}"

do

if [ "$optimizer" == "SGD" ]

then

for m in "${MOMENTUM_RATES[@]}"

do

echo "Network: " $network "Optimizer: " $optimizer "Learning rate: " $lr "Momentum: " $m  "Activation function: " $activation "Dropout: " $dropout "Epochs: " $epochs 
python recsys_fnn.py $network $lr $optimizer $activation $dropout $epochs $m $dataset $split > "temp"$split".txt"
tail -n 1 "temp"$split".txt" >> $filename

tail -n 1 $filename

done

else

m="0"

echo "Network: " $network "Optimizer: " $optimizer "Learning rate: " $lr  "Activation function: " $activation "Dropout: " $dropout "Epochs: " $epochs

python recsys_fnn.py $network $lr $optimizer $activation $dropout $epochs $m $dataset $split > "temp"$split".txt"
tail -n 1 "temp"$split".txt" >> $filename

tail -n 1 $filename

fi

done

done

done
		
done

done

grep -v Invalid $filename > "clean_"$filename
 
#done

echo "Done!"

