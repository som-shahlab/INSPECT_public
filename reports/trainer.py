"""
Clinical Document Classification 

- Evalute different document-level classification models

Adapted from https://github.com/luoyuanlab/Clinical-Longformer/blob/main/classification/openI_clinical_longformer.ipynb
@author Jason Alan Fries
@date 2022-11-02

Example Usage

python trainer.py \
--train data/pulmonary_embolisms/refactor_2022_11_05/debug_train.tsv \
--valid data/pulmonary_embolisms/refactor_2022_11_05/debug_valid.tsv \
--pretrained yikuan8/Clinical-Longformer \
--batch_size 8 \
--max_len 1024 \
--n_epochs 4 \
--label_key pe_acute,pe_subsegmentalonly,pe_positive,pe_uncertain \
--device cpu \
--outputdir /Users/jfries/Desktop/debug/


python trainer.py \
--train data/pulmonary_embolisms/refactor_2022_11_05/train.tsv \
--valid data/pulmonary_embolisms/refactor_2022_11_05/valid.tsv \
--pretrained /local-scratch-nvme/nigam/huggingface/pretrained/Clinical-Longformer \
--batch_size 4 \
--max_len 1280 \
--label_key pe_acute,pe_subsegmentalonly,pe_positive \
--device cuda:0 \
--n_epochs 4 \
--outputdir /local-scratch/nigam/projects/jfries/output/


"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


import torch
import argparse
import pandas as pd
import numpy as np
from torch import nn
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
)
import numpy as np
from sklearn.metrics import classification_report
from functools import partial
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from collections import defaultdict
from pathlib import Path

# transformers.logging.set_verbosity_error()


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))


def compute_metrics(eval_preds, label_names=["positive", "negative"]):
    logits, labels = eval_preds
    outputs = sigmoid(logits)
    y_pred = np.round(outputs)
    print("metric", logits.shape, labels.shape)
    return classification_report(
        labels, y_pred, target_names=label_names, output_dict=True
    )


class NotesDataset(torch.utils.data.Dataset):
    """Simple wrapper for a document + multuple labels"""

    def __init__(self, encodings, labels, num_labels):
        self.encodings = encodings
        self.labels = labels
        self.num_labels = num_labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.num_labels == 1:
            item["labels"] = torch.tensor(self.labels[idx][0]).float()
        else:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()

        # print(logits.view(-1, self.model.config.num_labels))
        # print(labels.float().view(-1, self.model.config.num_labels))

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


class SinglelabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()

        # print(logits.view(-1, self.model.config.num_labels))
        # print(labels.float().view(-1, self.model.config.num_labels))

        outputs = outputs.view(-1, self.model.config.num_labels)

        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.float().view(-1, self.model.config.num_labels),
        )
        return (loss, outputs) if return_outputs else loss


def main(args):
    # load pretrained tokenizers + model
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_tokenizer)
    labels = args.label_key

    # training arguments
    training_args = TrainingArguments(
        output_dir=args.outputdir,
        report_to="wandb",
        num_train_epochs=args.n_epochs,
        auto_find_batch_size=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        optim=args.optimizer,
        dataloader_num_workers=8,
    )

    if args.fp16:
        training_args.half_precision_backend = "amp"
        training_args.fp16 = args.fp16

    print("\n")
    print("=" * 80)
    print(f"number of labels: {len(labels)}")
    print("=" * 80)
    print("\n")

    # if len(labels) == 1:
    #    num_labels = 2
    # else:
    num_labels = len(labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.pretrained, num_labels=num_labels
    )

    # if ',' in args.device_ids:
    #    device_ids = [int(ids) for ids in args.device_ids.split(',')]
    #    model = nn.DataParallel(model, device_ids)
    # else:
    model = model.to(device)

    if args.train is not None:
        if args.valid is None:
            raise Exception("Validation dataset required during training")

        # load raw data
        train_data = pd.read_csv(args.train, sep=args.sep)
        valid_data = pd.read_csv(args.valid, sep=args.sep)

        # train_data = train_data[train_data[labels[0]] != "Censored"]
        print(train_data[labels].value_counts())

        # valid_data = valid_data[valid_data[labels[0]] != "Censored"]
        print(valid_data[labels].value_counts())

        # print dataset summary
        print(f"train: {len(train_data)}")
        print(f"valid: {len(valid_data)}")

        # apply tokenizer
        train_enc = tokenizer(
            train_data[args.text_key].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_len,
        )
        valid_enc = tokenizer(
            valid_data[args.text_key].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_len,
        )

        # create dataset wrapper for pytorch
        train_dataset = NotesDataset(
            train_enc, train_data[labels].values, num_labels=num_labels
        )
        valid_dataset = NotesDataset(
            valid_enc, valid_data[labels].values, num_labels=num_labels
        )
        if num_labels == 1:
            trainer = SinglelabelTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                compute_metrics=partial(compute_metrics, label_names=labels),
            )

        results = trainer.train()
        print("RESULTS", results.metrics)

    if args.test is not None:
        # load raw data
        test_data = pd.read_csv(args.test, sep=args.sep)

        # test_data = test_data[test_data[labels] != "Censored"]
        print(test_data[labels].value_counts())

        # print dataset summary
        print(f"test: {len(test_data)}")
        # apply tokenizer
        test_enc = tokenizer(
            test_data[args.text_key].tolist(),
            truncation=True,
            padding=True,
            max_length=args.max_len,
        )

        # create dataset wrapper for pytorch
        test_dataset = NotesDataset(
            test_enc, test_data[labels].values, num_labels=num_labels
        )

        if num_labels == 1:
            trainer = SinglelabelTrainer(
                model=model,
                args=training_args,
                compute_metrics=compute_metrics,
            )
        else:
            trainer = MultilabelTrainer(
                model=model,
                args=training_args,
                compute_metrics=partial(compute_metrics, label_names=labels),
            )

        results = trainer.predict(test_dataset)
        print("RESULTS", results.metrics)

        # extract prediction probability
        probs = torch.sigmoid(torch.tensor(results.predictions)).tolist()

        # convert prediction prob to dataframe
        results = defaultdict(list)
        for p in probs:
            for idx, l in enumerate(labels):
                results[f"{l}_prob"].append(p[idx])
        results_df = pd.DataFrame.from_dict(results)

        print(results_df.shape)
        print(test_data.shape)

        # save results
        output_dir = Path(args.outputdir)
        results_path = output_dir / args.result_file.replace(
            ".csv", f"_{args.test.split('_')[-2]}.csv"
        )
        results_df = pd.concat([test_data, results_df], axis=1)

        # binarize predictions
        LABELS = args.label_key
        THRESHOLD = 0.5
        for l in LABELS:
            results_df[f"{l}_pred"] = results_df[f"{l}_prob"].apply(
                lambda x: 1 if x >= THRESHOLD else 0
            )

        results_df.to_csv(results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, help="training set", default=None)
    parser.add_argument("--valid", type=str, help="validation set", default=None)
    parser.add_argument("--test", type=str, help="test set", default=None)
    parser.add_argument("--outputdir", type=str, help="output directory", required=True)

    parser.add_argument("--sep", type=str, default="\t", help="dataset delimiter")
    parser.add_argument("--text_key", type=str, default="text", help="text column key")
    parser.add_argument(
        "--label_key", type=str, default="text", help="label column keys", required=True
    )

    parser.add_argument(
        "--pretrained", type=str, help="Hugging Face pretrained model", required=True
    )
    parser.add_argument(
        "--pretrained_tokenizer",
        type=str,
        help="Hugging Face pretrained model",
        required=True,
    )
    parser.add_argument("--max_len", type=int, default=1024, help="max token length")
    parser.add_argument("--device", type=str, default="cpu", help="PyTorch device")

    parser.add_argument("--n_epochs", type=int, default=1, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="learning rate")
    parser.add_argument(
        "--optimizer", type=str, default="adamw_torch", help="optimizer"
    )
    parser.add_argument("--device_ids", type=str, default="0")
    parser.add_argument("--result_file", type=str, default="results.csv")

    parser.add_argument("--fp16", action="store_true", help="fp16")

    args = parser.parse_args()

    # if torch.backends.mps.is_available() and args.device == "mps":
    #    print("Using MPS (Apple Silicon GPU)")
    device = torch.device(args.device)

    # mixed precision
    if args.fp16 and "cuda" not in args.device:
        args.fp16 = False
        print("disabling fp16, requires CUDA device")

    if "," in args.label_key:
        args.label_key = args.label_key.split(",")
    else:
        args.label_key = [args.label_key]

    main(args)
