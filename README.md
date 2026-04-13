## Author
Woosuk Byun 

## Project Structure
    logreg_framework
        -data.py
        -main.py
        -model.py
        -train.py
        -utils.py

## Train Data Format
   X1 truthlabel1
   X2 truthlabel2
   .
   .
   .
   X3 truthlabeln

## Test Data Format
   X1 truthlabel1
   X2 truthlabel2
   .
   .
   .
   X3 truthlabeln


## How to train via CLI (Command Line Interface)
    example: python main.py --mode train --data train.txt --model model.npz --epochs 1 --lr 0.01

Arguments:
--mode train     : run training mode
--data           : path to training dataset
--model          : file to save trained model
--epochs         : number of epochs
--lr             : learning rate

## How to predict via CLI (Command Line Interface)
    example: python main.py --mode predict --data test.txt --model model.npz

Arguments:
--mode predict   : run prediction mode
--data           : path to test data for prediction
--model          : path to trained model file (.npz)
