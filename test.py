import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict
from datasets import BaseDataset
import pandas as pd

from model import *

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION,
    MODEL_ORIGINER,
    init_logger,
    set_seed,
    compute_metrics
)
from processor import seq_cls_load_and_cache_examples as load_and_cache_examples
from processor import seq_cls_tasks_num_labels as tasks_num_labels
from processor import seq_cls_processors as processors
from processor import seq_cls_output_modes as output_modes

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running Test on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running Test on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    txt_all = []

    for batch in progress_bar(eval_dataloader):
        model.eval()
        txt_all.append(batch[4])
        print(txt_all)
        batch = batch[:-1]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3]
            }
            if args.model_type not in ["distilkobert", "xlm-roberta"]:
                inputs["token_type_ids"] = batch[2]  # Distilkobert, XLM-Roberta don't use segment_ids


            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if output_modes[args.task] == "classification":
        preds = np.argmax(preds, axis=1)
    elif output_modes[args.task] == "regression":
        preds = np.squeeze(preds)
    result = compute_metrics(args.task, out_label_ids, preds)
    results.update(result)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir,
                                    "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return preds, out_label_ids, results, txt_all


def main(cli_args):
    # Read from config file and make args
    args = torch.load(os.path.join("ckpt",cli_args.result_dir,"checkpoint-10","training_args.bin"))
    logger.info("Testing parameters {}".format(args))
    
    checkpoints = list(
            os.path.dirname(c) for c in
            sorted(glob.glob(os.path.join(args.ckpt_dir, cli_args.result_dir)+"/**/"+"training_model.bin"))
        )
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    init_logger()

    processor = processors[args.task](args)
    labels = processor.get_labels()
    if output_modes[args.task] == "regression":
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task]
        )
    else:
        config = CONFIG_CLASSES[args.model_type].from_pretrained(
            args.model_name_or_path,
            num_labels=tasks_num_labels[args.task],
            id2label={str(i): label for i, label in enumerate(labels)},
            label2id={label: i for i, label in enumerate(labels)},
        )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    args.device = "cuda:{}".format(cli_args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu"
    

    # Load dataset
    test_dataset = BaseDataset(args, tokenizer, mode="test") if args.test_file else None

    for checkpoint in checkpoints:
        logger.info("Testing model checkpoint to {}".format(checkpoint))
        global_step = checkpoint.split("-")[-1]
        model = MODEL_LIST[cli_args.model_mode](args.model_type, args.model_name_or_path, config)
        model.load_state_dict(torch.load(checkpoint+"/training_model.bin"))
        model.to(args.device)
        preds, labels, result, txt_all = evaluate(args, model, test_dataset, mode="test", global_step=global_step)

        pred_and_labels = pd.DataFrame([])
        pred_and_labels["data"] = txt_all
        pred_and_labels["pred"] = preds
        pred_and_labels["label"] = labels
        pred_and_labels["result"] = preds==labels

        pred_and_labels.to_csv(os.path.join(args.ckpt_dir, cli_args.result_dir,checkpoint)+"/test_result.csv")

            


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default="config")
    cli_parser.add_argument("--config_file", type=str, required=True)
    cli_parser.add_argument("--result_dir", type=str, required=True)
    cli_parser.add_argument("--model_mode", type=str, required=True, choices=MODEL_LIST.keys())
    cli_parser.add_argument("--gpu", type=str, required=True)

    cli_args = cli_parser.parse_args()

    main(cli_args)
