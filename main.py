import argparse
from model import LogisticRegression
from train import train
from data import load_txt
from utils import save_model,load_model

def main():
    parser = argparse.ArgumentParser(description="Logistic Regression Framework")
    parser.add_argument("--mode",choices=["train","predict"],required=True)
    parser.add_argument("--data",required=True)
    parser.add_argument("--model",required=True)
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--lr",type=float,default=0.01)

    args = parser.parse_args()
    X,y = load_txt(args.data)

    if args.mode == "train":
       model = LogisticRegression(X.shape[1])
       train(model,X,y,args.lr,args.epochs)
       save_model(model,args.model)
       print("model saved!")

    if args.mode == "predict":
        model = load_model(args.model)
        predictions = model.predict(X)
        print("predictions for each test sample...")
        for p in predictions:
            print(p[0])

    
main()
