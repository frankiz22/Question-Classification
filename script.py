import argparse
import os
from model import PredictModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor CLI")
    subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")

    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--text", help="Input text for prediction")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate predictions")
    evaluate_parser.add_argument("--path_to_dataset", help="Path to the dataset file")

    args = parser.parse_args()
    
    model_path = 'results/camemberta_base/checkpoint-500'

    
    
    classes = ['book_flight', 'book_hotel', 'carry_on', 'flight_status',
               'lost_luggage', 'translate', 'travel_alert',
               'travel_suggestion', 'out_of_scope']

    
    
    print("-------------------loading model-----------------------")
    model = PredictModel(model_path)

    if args.subcommand == "predict":
        print("-------------------prediction-----------------------")
        prediction = model.predict(args.text, classes)
        print(f"Predicted class for input '{args.text}': {prediction}")
        
    elif args.subcommand == "evaluate":
        output_path = './data/eval/'
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("-------------------evaluation-----------------------")
        evaluation_results = model.evaluate_dataset(args.path_to_dataset,output_path,  classes)
        print("Evaluation Results:")
        print(evaluation_results)

