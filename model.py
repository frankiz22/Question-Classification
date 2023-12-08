from transformers import (AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, DataCollatorWithPadding,
                            TrainingArguments, AutoModelForSequenceClassification)
from datasets import Dataset
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
import json
import torch.nn as nn
import evaluate
from evaluate import evaluator
from typing import List


def compute_metrics(eval_preds):
    metrics = {"accuracy": evaluate.load("accuracy"),
               "precision": evaluate.load("precision"),
               "recall": evaluate.load("recall"),
               "f1": evaluate.load("f1")}
    
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    scores = {}
    for name, metric in metrics.items():
        if name != 'accuracy':
            current = metric.compute(predictions=predictions, references=labels, average = 'macro')
            scores[name + "_macro"] = current[name]
            current = metric.compute(predictions=predictions, references=labels, average = 'micro')
            scores[name + "_micro"] = current[name]
        else:
            current = metric.compute(predictions=predictions, references=labels)
            scores["accuracy"] = current["accuracy"]
        
    return scores
    
class PredictModel(AutoModelForSequenceClassification):
    def __init__(self, model_path):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        

    def predict(self , text: str, classes: List[str]):
        """
        Classify the text according to classes.
    
        Args:
        - classes (List[str]): List of classes for the prediction.
        - input_text (str): Input text to generate predictions.
    
        Returns:
        - str: Prediction class for text.
        """
        reverse_mapping = {idx:classe for idx,classe in enumerate(classes)}
        device = self.model.device
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        logits = outputs.get("logits")
        pred = torch.argmax(logits.view(-1, self.model.config.num_labels)).item()
        pred = reverse_mapping[pred]
        
        return pred

    def preprocessing(self, example):
        return self.tokenizer(example["text"], truncation=True)
        
    def evaluate_dataset(self, path_to_dataset: str, path_to_save: str, classes: dict) -> dict:
        """
        Evaluate a dataset using a trained model.
    
        Args:
        - path_to_dataset (str): Path to the dataset for evaluation.
        - classes (dict): A dictionary mapping original labels to new labels.
        - path_to_save (str): Path to save the evaluation results.
    
        Returns:
        - dict: Evaluation results containing metrics and scores.
        """
        #path_to_save = './data/eval'
        mapping = {classe: idx for idx,classe in enumerate(classes)} 
        X_test = pd.read_csv(path_to_dataset)
        X_test['label'] = X_test.apply(lambda x: mapping[x["label"]], axis = 1)
    
        test_set = Dataset.from_pandas(X_test)
        test_set = test_set.map(self.preprocessing, batched=True)
    
        task_evaluator = evaluator("text-classification")
        
        training_args = TrainingArguments(
            output_dir=path_to_save, 
            per_device_eval_batch_size=8        
        )
        trainer = Trainer(
            self.model,
            training_args,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics
        )
           
        return trainer.evaluate(eval_dataset=test_set)