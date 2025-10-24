import os
import sys
import wandb
import argparse
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from peft import get_peft_model, LoraConfig, PeftModel
from transformers import TrainingArguments, EarlyStoppingCallback
from datasets import load_from_disk, concatenate_datasets
from transformers import Trainer
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from dataclasses import dataclass
from transformers import BitsAndBytesConfig
from datetime import datetime
from tabulate import tabulate

import wandb
from wandb.apis.public import Run

# Enable TF32 for better performance on supported GPUs
torch.set_float32_matmul_precision('high')  # Fix TF32 warning

@dataclass
class Counter:
    ok: int = 0
    TP: int = 0
    TN: int = 0
    FP: int = 0
    FN: int = 0
    total: int = 0


def compute_metrics(TP, FP, TN, FN):
    total = TP + FP + TN + FN
    accuracy = (TP + TN) / total if total > 0 else 0
    # loss = 1-accuracy
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP) if TP + FP > 0 else 0
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN) if TP + FN > 0 else 0
    # specificity = TN / (TN + FP)
    if recall + precision == 0:
        f1 = 0
    else:
        f1 = 2 * (recall * precision) / (recall + precision) if recall + precision > 0 else 0

    if TP + FN == 0:
        w_pos = 0
    else:
        w_pos = TP / (TP + FN) if TP + FN > 0 else 0
    if TN + FP == 0:
        w_neg = 0
    else:
        w_neg = TN / (TN + FP) if TN + FP > 0 else 0
    weighted_accuracy = (w_pos + w_neg) / 2
    # return accuracy, precision, recall, f1, weighted_accuracy, total
    # Weighted F1-score (binary classification: positive and negative classes)
    # Positive class F1 (same as f1 above)
    f1_pos = f1
    # Negative class F1
    precision_neg = TN / (TN + FN) if TN + FN > 0 else 0
    recall_neg = TN / (TN + FP) if TN + FP > 0 else 0
    f1_neg = 2 * (precision_neg * recall_neg) / (precision_neg + recall_neg) if precision_neg + recall_neg > 0 else 0
    # Support for each class
    support_pos = TP + FN
    support_neg = TN + FP
    total_support = support_pos + support_neg
    # Weighted F1-score
    weighted_f1 = ((f1_pos * support_pos) + (f1_neg * support_neg)) / total_support if total_support > 0 else 0

    return accuracy, precision, recall, f1, weighted_accuracy, weighted_f1, total

def counter_update_ok(counter):
    counter.ok = counter.TP + counter.TN


def counter2str(counter):
    return f"{counter.total}\t{counter.TP}\t{counter.TN}\t{counter.FP}\t{counter.FN}"


def collate_fn(examples, prompt):
    texts = ["<image>" + prompt + example["question"] for example in examples]
    labels = [example["answer"] for example in examples]
    images = [example["image"].convert("RGB") for example in examples]
    tokens = processor(text=texts, images=images, suffix=labels,
                      return_tensors="pt", padding="longest")
    tokens = tokens.to(DTYPE).to(device)
    return tokens


def evaluate_model(dataset, dataset_name, prompt, model, processor, device, dtype, max_padding_length):
    model.eval()
    predictions = []
    true_labels = []

    print(f"\nEvaluating {dataset_name}...")

    with torch.no_grad():
        for example in tqdm(dataset):
            text = "<image>" + prompt + example["question"]
            image = example["image"].convert("RGB")
            true_label = example["answer"]

            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,  # Enable padding
                max_length=max_padding_length,  # Adjust based on your dataset
                truncation=True,  # Truncate long inputs
                return_attention_mask=True  # Ensure attention mask is included
            ).to(device)
            inputs = {k: v.to(dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                # temperature=0.0,
                # use_cache=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            generated_text = generated_text.strip().upper()
            pred_label = "Yes" if "YES" in generated_text else "No" if "NO" in generated_text else "Unknown"

            predictions.append(pred_label)
            true_labels.append(true_label)

    print(f"\n{dataset_name} Classification Report:")
    report = classification_report(true_labels, predictions, target_names=["No", "Yes"], output_dict=True)
    print(classification_report(true_labels, predictions, target_names=["No", "Yes"]))

    # Log metrics to W&B
    metrics = {
        f"{dataset_name.lower()}_accuracy": accuracy_score(true_labels, predictions),
        f"{dataset_name.lower()}_precision": report["weighted avg"]["precision"],
        f"{dataset_name.lower()}_recall": report["weighted avg"]["recall"],
        f"{dataset_name.lower()}_f1": report["weighted avg"]["f1-score"],
    }
    wandb.log(metrics)

    return predictions, true_labels


def evaluate_model_per_query_type(dataset, dataset_name, prompt, model, processor, device, dtype, max_padding_length):
    model.eval()

    counters = {
        "clef": Counter(),
        "meter": Counter(),
        "pitch": Counter(),
        "pitch_no_scale": Counter(),
        "interval": Counter(),
        "degree": Counter(),
        "rythm": Counter(),
        "melody": Counter()
    }
    total_counter = Counter()

    TP = 0
    FP = 0
    TN = 0
    FN = 0

    with torch.no_grad():

        for example in tqdm(dataset):
            text = "<image>" + prompt + example["question"]
            image = example["image"].convert("RGB")
            true_label = example["answer"]
            query_type = example["query_type"]

            inputs = processor(
                text=text,
                images=image,
                return_tensors="pt",
                padding=True,  # Enable padding
                max_length=max_padding_length,  # Adjust based on your dataset
                truncation=True,  # Truncate long inputs
                return_attention_mask=True  # Ensure attention mask is included
            ).to(device)
            inputs = {k: v.to(dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}

            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                # temperature=0.0,
                # use_cache=False,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            pred_label = "YES" in generated_text.strip().upper()
            true_label = "YES" in true_label.upper()

            counters[query_type].total += 1

            if pred_label and true_label:
                TP += 1
                counters[query_type].TP += 1
                counters[query_type].ok += 1
            elif pred_label and not true_label:
                FP += 1
                counters[query_type].FP += 1
            elif not pred_label and not true_label:
                TN += 1
                counters[query_type].TN += 1
                counters[query_type].ok += 1
            else:
                FN += 1
                counters[query_type].FN += 1

        for counter in counters.values():
            total_counter.total += counter.total
            total_counter.ok += counter.ok

    # log global metrics
    accuracy, precision, recall, f1, weighted_accuracy, weighted_f1, total = compute_metrics(TP, FP, TN, FN)
    metrics = {
        f"{dataset_name.lower()}/TP": TP,
        f"{dataset_name.lower()}/TN": TN,
        f"{dataset_name.lower()}/FP": FP,
        f"{dataset_name.lower()}/FN": FN,
        f"{dataset_name.lower()}/accuracy": accuracy,
        f"{dataset_name.lower()}/weighted_accuracy": weighted_accuracy,
        f"{dataset_name.lower()}/precision": precision,
        f"{dataset_name.lower()}/recall": recall,
        f"{dataset_name.lower()}/f1": f1,
        f"{dataset_name.lower()}/total": total,
    }
    wandb.log(metrics)

    # Create a wandb table for metrics
    table = wandb.Table(columns=[
        "Modality", "TP", "TN", "FP", "FN", "Accuracy", "Precision", "Recall", "F1", "Weighted_Accuracy",
        "Weighted_F1", "Total"
    ])

    # Compute and log global metrics
    accuracy, precision, recall, f1, weighted_accuracy, weighted_f1, total = compute_metrics(
        TP, FP, TN, FN
    )
    table.add_data(
        "Global", TP, TN, FP, FN,
        round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4),
        round(weighted_accuracy, 4), round(weighted_f1, 4), total
    )

    # Compute and log metrics for each query_type
    for counter_type, counter in counters.items():
        c_accuracy, c_precision, c_recall, c_f1, c_weighted_accuracy, c_weighted_f1, c_total = compute_metrics(
            counter.TP, counter.FP, counter.TN, counter.FN
        )
        table.add_data(
            counter_type, counter.TP, counter.TN, counter.FP, counter.FN,
            round(c_accuracy, 4), round(c_precision, 4), round(c_recall, 4), round(c_f1, 4),
            round(c_weighted_accuracy, 4), round(c_weighted_f1, 4), c_total
        )

    # Log the table to wandb
    wandb.log({f"{dataset_name.lower()}/metrics_table": table})


def find_latest_checkpoint(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    nums = [int(d.split("-")[1]) for d in checkpoints]
    latest_num = max(nums)
    return os.path.join(output_dir, f"checkpoint-{latest_num}")


def find_run_id(project_name, experiment_name):
    api = wandb.Api()
    runs = api.runs(f"ffuhu/{project_name}")
    for run in runs:
        if run.name == experiment_name:
            return run.id
    print(f"No run found with name {experiment_name} in project {project_name}")
    return None


if __name__ == "__main__":

    # Enable verbose logging
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    os.environ["TORCH_LOGS"] = "recompiles"

    # Argument parser
    parser = argparse.ArgumentParser(description="Train or evaluate a PaliGemma model with PEFT")
    parser.add_argument("--exp_desc", type=str, default="stg1", help="Experiment description")
    parser.add_argument("--dataset_name", type=str, default="fmt-c", help="Name of the dataset")
    parser.add_argument("--model_size", type=str, default="3B", help="Size of the paligemma2 model (3B, 10B or 28B)")
    parser.add_argument("--resolution", type=int, default=224, help="Resolution of the paligemma2 model (3B, 10B or 28B)")

    parser.add_argument("--train_vision_tower", type=bool, default=True, help="Whether to train the vision tower")
    parser.add_argument("--train_mm_projector", type=bool, default=True,
                        help="Whether to train the multi-modal projector")
    parser.add_argument("--use_lora", type=bool, default=True, help="Whether to use LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--quantization_bits", type=int, default=None, help="Quantization bits (4, 8) or None")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--prompt_type", type=str, default="text", help="Type of prompt (e.g., text)")
    # parser.add_argument("--model_id", type=str, default="google/paligemma2-3b-pt-224", help="Model identifier")
    parser.add_argument("--prompt", type=str,
                        default="Is this kern notation present in the musical staff in the image: ", help="Prompt text")
    # parser.add_argument("--max_padding_length", type=int, default=256 + 64, help="Maximum padding length for sequences")
    parser.add_argument("--output_folder", type=str, default="outputs", help="Output directory")


    parser.add_argument("--debug", type=bool, default=False, help="Debug uses a tiny fraction of the dataset")
    parser.add_argument("--dry-run", type=bool, default=False, help="To check the parameters are ok")

    args = parser.parse_args()

    # Initialize W&B project
    wandb.login()  # Ensure you have W&B API key set in environment or prompted here

    # Fixed params for Stage 1
    exp_desc = args.exp_desc  #"stg1"
    dataset_name = args.dataset_name  #"fmt-c"
    train_vision_tower = args.train_vision_tower  #True
    train_mm_projector = args.train_mm_projector  #True
    use_lora = args.use_lora  #True
    lora_rank = args.lora_rank  #8
    batch_size = args.batch_size  #1
    gradient_accumulation_steps = args.gradient_accumulation_steps  #4
    weight_decay = args.weight_decay  #1e-6
    warmup_ratio = args.warmup_ratio  #0.1
    epochs = args.epochs  #5
    prompt_type = args.prompt_type  #"text"
    # model_id = args.model_id  #"google/paligemma2-3b-pt-224"
    prompt = args.prompt  #"Is this kern notation present in the musical staff in the image: "
    # max_padding_length = args.max_padding_length  #256 + 64
    exp_id = datetime.now().strftime("%d%m%y%H%M%S")
    learning_rate = args.learning_rate
    train_vision_tower = args.train_vision_tower
    use_lora = args.use_lora
    lora_rank = args.lora_rank
    quantization_bits = args.quantization_bits

    # show arguments
    print(f"\n{'=' * 50}")
    print(f"EVALUATING Experiment {exp_id}")
    print(f"\nRun started at: {datetime.now().strftime('%d%m%y%H%M%S')}")
    print("\nArguments:")
    args_table = [[key, value] for key, value in vars(args).items()]
    print(tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print(f"{'=' * 50}")

    if args.dry_run:
        print("Dry run")
        sys.exit(0)

    model_id = f"google/paligemma2-{args.model_size}-pt-{args.resolution}"
    max_padding_length = 256 + 64 if args.resolution == 224 else 1024 + 64 if args.resolution == 448 else 4096 + 64
    lora_info = f"LoRA{use_lora}-{lora_rank}" if use_lora else f"LoRA{use_lora}"
    quantization_info = f"quant{quantization_bits}" if quantization_bits else "quantNo"

    # for testing, cannot keep the date, because it will not match
    experiment_name = (f"{dataset_name}_"
                       f"{model_id.replace('/', '-')}_"
                       f"trainVT{train_vision_tower}_"
                       f"trainMMP{train_mm_projector}_"
                       f"lr{learning_rate}_"
                       f"bs{batch_size}_"
                       f"wd{weight_decay}_"
                       f"wur{warmup_ratio}_"
                       f"{lora_info}_"
                       f"{quantization_info}"
                       f"epochs{epochs}_"
                       f"prompt{prompt_type}")

    # find the experiment
    experiment_name = [path for path in os.listdir(args.output_folder) if experiment_name in path]
    if len(experiment_name) > 1:
        print(f"WARNING: Multiple experiments found for {dataset_name}")
        print(f"Picking the last one:"
              f"\n\t{experiment_name[-1]}")
    experiment_name = experiment_name[-1]

    output_dir = f"./{args.output_folder}/{experiment_name}"

    # Find latest checkpoint
    checkpoint_path = find_latest_checkpoint(output_dir)
    if checkpoint_path is None:
        print("No checkpoint found in", output_dir)
        sys.exit(1)
    print(f"Loading from checkpoint: {checkpoint_path}")

    # Processor loaded once
    processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)

    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # find the wandb run_id to append to it
    run_id = find_run_id("paligemma2-vqa", experiment_name)
    resume = "must" if run_id else "allow"
    print(f"Run ID: {run_id}")

    # Initialize W&B run
    wandb.init(
        project="paligemma2-vqa",
        group="stage1",
        name=experiment_name,
        config={
            "model_id": model_id,
            "learning_rate": learning_rate,
            "train_vision_tower": train_vision_tower,
            "train_mm_projector": train_mm_projector,
            "use_lora": use_lora,
            "lora_rank": lora_rank,
            "quantization_bits": quantization_bits,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "epochs": epochs,
            "prompt_type": prompt_type,
            "dataset": dataset_name,
            "max_padding_length": max_padding_length,
        },
        id=run_id,
        resume=resume,
        # config=var(args),
    )

    # Quantization config
    quant_config = None
    if args.quantization_bits == 4:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
    elif args.quantization_bits == 8:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)

    load_kwargs = {"quantization_config": quant_config} if quant_config else {"torch_dtype": dtype}

    # Load model
    is_peft = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    if is_peft:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", **load_kwargs
        )
        model = PeftModel.from_pretrained(model, checkpoint_path)
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            checkpoint_path, device_map="auto", **load_kwargs
        )
    model.eval()
    DTYPE = model.dtype


    ood_datasets = ["fmt-c", "fmt-m", "malaga", "primusn", "all"]
    ood_datasets = ["all", "primusn"]
    #ood_datasets = ["all"]
    for ood_ds_name in ood_datasets:

        # if dataset_name == ood_ds_name:
        #     continue

        # Load datasets once
        dataset_path = f"./datasets/{ood_ds_name}-vqa"
        # train_ds = load_from_disk(os.path.join(dataset_path, 'train'))
        # val_ds = load_from_disk(os.path.join(dataset_path, 'val'))
        test_ds = load_from_disk(os.path.join(dataset_path, 'test'))

        if ood_ds_name == "all":
            # Load datasets once
            all_ds_names = ["fmt-c", "fmt-m", "malaga", "primusn"]
            test_dss = {}
            for ds_name in all_ds_names:
                dataset_path = f"./datasets/{ds_name}-vqa"
                test_dss[ds_name] = load_from_disk(os.path.join(dataset_path, 'test'))

            # combine all datasets
            test_ds = concatenate_datasets([*test_dss.values()])

        # for debug only
        if args.debug:
            print("DEBUGGING WITH VERY LITTLE DATA!!!")
            exp_desc = "DEBUGGING_"
            # train_ds = train_ds.train_test_split(test_size=0.995)["train"]  # we'll use a very small split for demo
            # val_ds = val_ds.train_test_split(test_size=0.99)["train"]  # we'll use a very small split for demo
            test_ds = test_ds.train_test_split(test_size=0.99)["train"]  # we'll use a very small split for demo

        # Final evaluations
        print("\n" + "=" * 50)
        print(f"EVALUACIÓN FINAL - Exp {exp_id}")
        print("=" * 50)

        # evaluate_model(train_ds, "Train Set", prompt, model, processor, device, DTYPE, max_padding_length)
        # evaluate_model(val_ds, "Val Set", prompt, model, processor, device, DTYPE, max_padding_length)
        # evaluate_model(test_ds, "Test Set", prompt, model, processor, device, DTYPE, max_padding_length)

        #evaluate_model_per_query_type(train_ds, f"{ood_ds_name}/Train Set", prompt, model, processor, device, DTYPE, max_padding_length)
        #evaluate_model_per_query_type(val_ds, f"{ood_ds_name}/Val Set", prompt, model, processor, device, DTYPE, max_padding_length)
        evaluate_model_per_query_type(test_ds, f"{ood_ds_name}/Test Set", prompt, model, processor, device, DTYPE, max_padding_length)

    # Finish W&B run
    wandb.finish()

    del model