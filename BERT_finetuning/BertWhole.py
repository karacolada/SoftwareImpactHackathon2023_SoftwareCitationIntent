from datasets import Dataset
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from transformers import *
from datasets import Dataset
import pandas as pd
import evaluate
from transformers import DataCollatorWithPadding
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import pipeline
import argparse
import torch


class Model:
    def __init__(self, model_name, id2label, label2id):
        self.model_name = model_name
        if model_name == 'bert':
            self.model_name = 'bert-base-uncased'
        elif model_name == 'distilbert':
            self.model_name = 'distilbert-base-uncased'
        elif model_name == 'pubmedbert':
            self.model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
        elif model_name == 'scibert':
            self.model_name = 'allenai/scibert_scivocab_uncased'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=4, id2label=id2label, label2id=label2id
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.id2label = id2label
        self.label2id = label2id
    def load_dataset(self, dataset):
        self.dataset = dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        def preprocess_function(examples):
            return self.tokenizer(examples["text"], truncation=True)
        self.tokenizer_dataset = self.dataset.map(preprocess_function, batched=True)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.accuracy = evaluate.load("accuracy")
    def train(self, training_args):
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return self.accuracy.compute(predictions=predictions, references=labels)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenizer_dataset["train"],
            eval_dataset=self.tokenizer_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=compute_metrics,
        )
        self.trainer.train()
    def get_classifier(self):
        classifier = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
        return classifier

    def evaluate(self, true_labels, predicted_labels):
        p, r, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, average='macro')
        accuracy = round(accuracy_score(true_labels, predicted_labels), 3)
        print('Precision: ', round(p, 3), 'Recall: ', round(r, 3), 'F1:', round(f1, 3), 'Accuracy:', accuracy)
        print(classification_report(true_labels, predicted_labels))
        

def main():
    parser = argparse.ArgumentParser(description='construct co-citation dict')
    parser.add_argument('--model', type=str, default='bert', choices=['bert', 'distilbert', 'pubmedbert', 'scibert'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()
    df = pd.read_csv('../data/software_citation_intent_merged.csv')
    dataset = df[['text', 'label']]
    dataset = dataset.sample(n=len(dataset), random_state=42).reset_index(drop=True)

    id2label = {0: "created", 1: "used", 2: "mention", 3: "none"}
    label2id = {"created": 0, "used": 1, "mention": 2, "none": 3}

    dataset = Dataset.from_pandas(dataset)
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    model = Model(args.model, id2label, label2id)
    model.load_dataset(dataset)

    training_args = TrainingArguments(
        output_dir="./tmp/",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        # load_best_model_at_end=True,
        save_strategy="no",
        # push_to_hub=True,
    )
    model.train(training_args)

    classifier = model.get_classifier()

    test = dataset['test']
    classifier_list = classifier([i['text'] for i in test])
    true_labels = [id2label[i['label']] for i in test]
    predicted_labels = [i['label'] for i in classifier_list]
    print("#"*40)
    print('test dataset')
    model.evaluate(true_labels, predicted_labels)

    ## validate on val dataset
    df_evaluate = pd.read_csv('../data/czi_val_merged.csv')
    df_evaluate = df_evaluate[['text', 'label']]
    text = df_evaluate['text'].values.tolist()
    map_label = {'usage':'used', 'creation':'created', 'none': 'none', 'mention': 'mention'}
    df_evaluate['label'] = df_evaluate['label'].apply(lambda x: map_label[x])
    classifier_list = classifier(text)

    true_labels = [i for i in df_evaluate['label'].values.tolist()]
    predicted_labels = [i['label'] for i in classifier_list]

    print("#"*40)
    print('validation dataset')
    evaluate(true_labels, predicted_labels)


if __name__ == '__main__':
    main()