import os
import sys
import pdb
import argparse
import torch
import numpy as np
import json
from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, Trainer, TrainingArguments
from torch.optim import AdamW
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
from os.path import join

from torch.nn.functional import cross_entropy
import mpi.hf_data_mpi as data_mpi


def preprocess_logits_for_metrics(logits, labels):
    # check the readme to see why this is made so oddly

    # print('preprocess')
    logits = logits[0]  # we get a random tuple with this idk why
    # print(logits.device) #cuda
    # print(logits.shape)
    # print(labels.shape)

    logits = logits.view([-1, logits.shape[-1]])
    labels = labels.view([-1])

    mask = labels != -100
    logits = logits[mask]
    labels = labels[mask]

    return cross_entropy(logits, labels, reduction='sum').reshape([1, 1])


def compute_metrics(eval_pred):
    cross = eval_pred.predictions[0]  # bad naming because of the hack
    return {'entropy': cross.sum()}


def main(args):
    with open(r'/home/nadavsc/LIGHTBITS/mpiricalplus/source/dataset/mpi.code-snippets', 'r') as f:
        file = json.load(f)
    extended_tokens = [prefix.lower() for prefix in file.keys()]

    model = GPTNeoXForCausalLM.from_pretrained(args.config)
    print('Model has been loaded')

    embedding_layer = model.get_input_embeddings()
    num_embeddings = embedding_layer.weight.shape[0]
    new_num_embeddings = num_embeddings+len(extended_tokens)
    model.resize_token_embeddings(new_num_embeddings)
    print(f'Embedding layer has changed: {num_embeddings} -> {new_num_embeddings}')

    if args.checkpoint and not args.resume:
        checkpoint = torch.load(join(args.checkpoint, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)

    train_dataloader, val_dataloader, test_dataloader = data_mpi.build_mpi_dataset(args)

    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps,
                      weight_decay=args.weight_decay)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                                   num_training_steps=args.training_steps)

    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=float('inf'),
        max_steps=args.training_steps,
        # num_train_epochs=args.epochs,
        # num_training_steps=args.training_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        eval_steps=args.eval_interval,
        logging_dir='./logs',
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=test_dataloader,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        optimizers=(optimizer, scheduler)
    )
    pdb.set_trace()

    if args.checkpoint and args.resume:
        print('resuming')
        trainer.train(args.checkpoint)
    else:
        trainer.train()
    trainer.evaluate()


if __name__ == '__main__':
    ckpt_home = '/home/nadavsc/LIGHTBITS/code-lms/polycoder/checkpoints/'
    ckpt_name = 'allc_tokom_700M'
    parser = argparse.ArgumentParser(description="Script to train GPT-2 model")

    parser.add_argument('--config', type=str, default=os.path.join(ckpt_home, ckpt_name), help="Path to the config for building the model")
    parser.add_argument('--save_dir', type=str, default='/home/nadavsc/LIGHTBITS/code-lms/polycoder/checkpoints/finetune', help="Directory to save model checkpoints")
    parser.add_argument('--batch_size', type=int, default=64, help="Big batch sizes are allowed (total #tokens per batch 262144)")

    parser.add_argument('--checkpoint', type=str, default=None, help="Huggingface checkpoint folder")
    parser.add_argument('--resume', action='store_true', default=False, help="whether to resume training or load the checkpoint")

    parser.add_argument('--lr', type=float, default=0.00016, help="Learning rate for the optimizer")
    # parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to train")
    parser.add_argument('--save_interval', type=int, default=1, help="Interval to save model checkpoints")
    parser.add_argument('--eval_interval', type=int, default=1, help="Interval to evaluate the model on test data")
    parser.add_argument('--warmup_steps', type=int, default=1600, help="Number of warmup steps")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument('--training_steps', type=int, default=150000, help="Total number of training steps")
    parser.add_argument('--adam_beta1', type=float, default=0.9, help="Beta1 for the Adam optimizer")
    parser.add_argument('--adam_beta2', type=float, default=0.999, help="Beta2 for the Adam optimizer")
    parser.add_argument('--adam_eps', type=float, default=1e-8, help="Epsilon for the Adam optimizer")

    parser.add_argument('-t', '--tokenizer_type', type=str, default='Tokompiler')
    parser.add_argument('-v', '--vocab_file', type=str,
                        default='../megatron/tokenizer/tokompiler/tokenizer_vocab/vocab.txt')
    parser.add_argument('-m', '--merge_file', type=str, default='../megatron/tokenizer/gpt_vocab/gpt2-merges.txt')
    parser.add_argument('-d', '--data_path', type=str, default='/home/nadavsc/LIGHTBITS/mpiricalplus/dataset/dataset_saved/tokompiler')
    parser.add_argument('--save', type=bool, default=True)
    # The following arguments are leftover from megatron settings -- you can keep the defaults
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--make_vocab_size_divisible_by', type=int, default=128)
    parser.add_argument('--model_parallel_size', type=int, default=1)


    args = parser.parse_args()
    main(args)