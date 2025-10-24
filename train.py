import os
import sys
import wandb
import argparse
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
from transformers import TrainingArguments, EarlyStoppingCallback
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset, load_from_disk, interleave_datasets
from transformers import Trainer
import torch
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
from dataclasses import dataclass
from transformers import BitsAndBytesConfig
from datetime import datetime
from tabulate import tabulate

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


class CustomSaveEvalCallback(TrainerCallback):
    def __init__(self, eval_steps, output_dir, save_total_limit, metric_for_best_model):
        self.eval_steps = eval_steps  # e.g., 20000 for every 20,000 steps
        self.output_dir = output_dir
        self.save_total_limit = save_total_limit
        self.metric_for_best_model = metric_for_best_model
        self.last_step_saved = -1  # Track last step where save occurred to avoid duplicates

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Check if current step is a multiple of eval_steps
        if state.global_step % self.eval_steps == 0 and state.global_step != self.last_step_saved:
            control.should_evaluate = True  # Trigger evaluation
            control.should_save = True      # Trigger save
            self.last_step_saved = state.global_step
        return control

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Trigger evaluation and save at epoch end, unless already saved at this step
        if state.global_step != self.last_step_saved:
            control.should_evaluate = True
            control.should_save = True
            self.last_step_saved = state.global_step
        return control

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Enforce save_total_limit by deleting old checkpoints
        checkpoints = sorted(
            [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1])
        )
        while len(checkpoints) > self.save_total_limit:
            oldest_checkpoint = os.path.join(self.output_dir, checkpoints.pop(0))
            import shutil
            shutil.rmtree(oldest_checkpoint)
        return control


class PaliGemmaTrainer(Trainer):
    def __init__(self, *args, processor=None, prompt="", max_padding_length=256+64, **kwargs):
        self.processor = processor
        self.prompt = prompt
        self.max_padding_length = max_padding_length
        super().__init__(*args, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return {}

        self.model.eval()
        predictions = []
        true_labels = []

        print(f"\n{'=' * 50}")
        print(f"Evaluating at epoch {self.state.epoch}...")
        print(f"{'=' * 50}")

        with torch.no_grad():
            for example in tqdm(dataset, desc="Evaluating"):
                text = "<image>" + self.prompt + example["question"]
                image = example["image"].convert("RGB")
                true_label = example["answer"]

                inputs = self.processor(
                    text=text,
                    images=image,
                    return_tensors="pt",
                    padding=True,  # Enable padding
                    max_length=self.max_padding_length,  # Adjust based on your dataset
                    truncation=True,  # Truncate long inputs
                    return_attention_mask=True  # Ensure attention mask is included
                ).to(self.model.device)
                inputs = {k: v.to(self.model.dtype) if v.dtype == torch.float32 else v for k, v in inputs.items()}

                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    # temperature=0.0,
                    # use_cache=False,
                    pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, 'tokenizer') else None
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                generated_text = generated_text.strip().upper()
                pred_label = "Yes" if "YES" in generated_text else "No" if "NO" in generated_text else "Unknown"

                predictions.append(pred_label)
                true_labels.append(true_label)

        print(f"\nClassification Report (Epoch {self.state.epoch}):")
        report = classification_report(true_labels, predictions, target_names=["No", "Yes"], output_dict=True)
        print(classification_report(true_labels, predictions, target_names=["No", "Yes"]))

        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        metrics = {
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_precision": precision,
            f"{metric_key_prefix}_recall": recall,
            f"{metric_key_prefix}_f1": f1,
        }

        # Log to W&B
        wandb.log(metrics)
        self.log(metrics)
        return metrics


def load(dataset_name, split, locally=False):
    if locally:
        print(f"Loading {dataset_name} locally...")
        dataset_path = f"./datasets/{args.dataset_name.lower()}-vqa"
        return load_from_disk(os.path.join(dataset_path, split))
    else:
        print(f"Loading {dataset_name} from HF...")
        dataset_path = f"PRAIG/vqa-{args.dataset_name.lower()}"
        return load_dataset(dataset_path, split)


if __name__ == "__main__":

    # Enable verbose logging
    os.environ["TRANSFORMERS_VERBOSITY"] = "info"
    os.environ["TORCH_LOGS"] = "recompiles"

    # Argument parser
    parser = argparse.ArgumentParser(description="Train or evaluate a PaliGemma model with PEFT")
    parser.add_argument("--exp_desc", type=str, default="stg1", help="Experiment description")
    parser.add_argument("--dataset_name", type=str, default="fmt-c", help="Name of the dataset")
    parser.add_argument("--load_dataset_locally", type=bool, default=False,
                        help="Whether or not to load the dataset locally")
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
    parser.add_argument("--max_steps", type=int, default=100_000,
                        help="Absolute maximum number of training steps (overrides number of epochs)")
    parser.add_argument("--eval_steps", type=int, default=20_000, help="Eval every eval_steps steps")
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

    # Set the experiment id to the date and time
    exp_id = datetime.now().strftime("%d%m%y_%H%M%S")

    # show arguments
    print(f"\n{'=' * 50}")
    print(f"Running Experiment {exp_id}")
    print("\nArguments:")
    args_table = [[key, value] for key, value in vars(args).items()]
    print(tabulate(args_table, headers=["Argument", "Value"], tablefmt="grid"))
    print(f"{'=' * 50}")

    if args.dry_run:
        print("Dry run")
        sys.exit(0)

    # Load datasets once
    all_ds_names = ["fmt-c", "fmt-m", "malaga", "primusn"]

    if args.dataset_name != "all":
        train_ds = load(args.dataset_name, 'train', args.load_dataset_locally)
        val_ds = load(args.dataset_name, 'val', args.load_dataset_locally)
        test_ds = load(args.dataset_name, 'test', args.load_dataset_locally)
    else:
        train_dss = {}
        val_dss = {}
        test_dss = {}
        for ds_name in all_ds_names:
            train_dss[ds_name] = load(ds_name, 'train', args.load_dataset_locally)
            val_dss[ds_name] = load(ds_name, 'val', args.load_dataset_locally)
            test_dss[ds_name] = load(ds_name, 'test', args.load_dataset_locally)

        # combine all datasets
        probabilities = [1/len(all_ds_names)] * len(all_ds_names)
        train_ds = interleave_datasets([*train_dss.values()], probabilities=probabilities, seed=42)
        val_ds = interleave_datasets([*val_dss.values()], probabilities=probabilities, seed=42)
        test_ds = interleave_datasets([*test_dss.values()], probabilities=probabilities, seed=42)

    # for debug only
    if args.debug:
        print("DEBUGGING WITH VERY LITTLE DATA!!!")
        args.exp_desc = "DEBUGGING_"
        train_ds = train_ds.train_test_split(test_size=0.995)["train"]  # we'll use a very small split for demo
        val_ds = val_ds.train_test_split(test_size=0.99)["train"]  # we'll use a very small split for demo
        test_ds = test_ds.train_test_split(test_size=0.99)["train"]  # we'll use a very small split for demo
        args.max_steps = 1_000
        args.eval_steps = 1_000

    print(f"Dataset size (train): {len(train_ds)}")
    print(f"Dataset size (val): {len(val_ds)}")
    print(f"Dataset size (test): {len(test_ds)}")

    model_id = f"google/paligemma2-{args.model_size}-pt-{args.resolution}"
    max_padding_length = 256 + 64 if args.resolution == 224 else 1024 + 64 if args.resolution == 448 else 4096 + 64
    lora_info = f"LoRA{args.use_lora}-{args.lora_rank}" if args.use_lora else f"LoRA{args.use_lora}"
    quantization_info = f"quant{args.quantization_bits}" if args.quantization_bits else "quantNo"
    experiment_name = (f"{args.exp_desc}exp{exp_id}_"
                       f"{args.dataset_name}_"
                       f"{model_id.replace('/', '-')}_"
                       f"trainVT{args.train_vision_tower}_"
                       f"trainMMP{args.train_mm_projector}_"
                       f"lr{args.learning_rate}_"
                       f"bs{args.batch_size}_"
                       f"wd{args.weight_decay}_"
                       f"wur{args.warmup_ratio}_"
                       f"{lora_info}_"
                       f"{quantization_info}"
                       f"epochs{args.epochs}_"
                       f"max_steps{args.max_steps}_"
                       f"prompt{args.prompt_type}")
    output_dir = f"./{args.output_folder}/{experiment_name}"

    # Processor loaded once
    processor = PaliGemmaProcessor.from_pretrained(model_id, use_fast=True)

    # Device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Initialize W&B run
    wandb.init(
        project="paligemma2-vqa",
        group="stage1",
        name=experiment_name,
        config=vars(args),
    )

    if args.quantization_bits == 4:
        # Define quantization config
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # Normalized Float 4 for better accuracy
            bnb_4bit_compute_dtype=torch.float16,  # Compute in FP16 for speed
            bnb_4bit_use_double_quant=True  # Nested quantization for extra memory savings
        )

    # Load model with quantization
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config if args.quantization_bits else None,
        device_map="auto"
    )

    if not args.train_vision_tower:
        for param in model.vision_tower.parameters():
            param.requires_grad = False

    if not args.train_mm_projector:
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

    if args.use_lora:
        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    DTYPE = model.dtype

    args_trainer = TrainingArguments(
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_torch_fused",
        # save_strategy="epoch",
        # eval_strategy="epoch",
        # save_strategy="steps",
        # eval_strategy="steps",
        save_strategy="no",  # CustomSaveEvalCallback takes care of it
        eval_strategy="no",
        # eval_steps=20000,
        # save_steps=20000,
        save_total_limit=2,
        output_dir=output_dir,
        bf16=torch.cuda.is_available(),
        report_to=["wandb"],  # Add W&B
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        label_names=["answer"],
    )


    # Instantiate the custom callback
    custom_callback = CustomSaveEvalCallback(
        eval_steps=args.eval_steps,  # Evaluate/save every 20,000 steps
        output_dir=output_dir,
        save_total_limit=2,
        metric_for_best_model="eval_f1"
    )


    trainer = PaliGemmaTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda examples: collate_fn(examples, args.prompt),
        args=args_trainer,
        processor=processor,
        prompt=args.prompt,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.005),
            custom_callback
        ],
        max_padding_length=max_padding_length,
    )

    trainer.train()

    # Final evaluations
    print("\n" + "=" * 50)
    print(f"EVALUACIÓN FINAL - Exp {exp_id}")
    print("=" * 50)

    # evaluate_model(train_ds, "Train Set", args.prompt, model, processor, device, DTYPE, max_padding_length)
    # evaluate_model(val_ds, "Val Set", args.prompt, model, processor, device, DTYPE, max_padding_length)
    # evaluate_model(test_ds, "Test Set", args.prompt, model, processor, device, DTYPE, max_padding_length)

    evaluate_model_per_query_type(train_ds, "Train Set", args.prompt, model, processor, device, DTYPE, max_padding_length)
    evaluate_model_per_query_type(val_ds, "Val Set", args.prompt, model, processor, device, DTYPE, max_padding_length)
    evaluate_model_per_query_type(test_ds, "Test Set", args.prompt, model, processor, device, DTYPE, max_padding_length)

    # Finish W&B run
    wandb.finish()

    del model, trainer