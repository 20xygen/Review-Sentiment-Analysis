import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding


BATCH_SIZES = (1, 2, 4, 8, 16)
REPEATS = 3
ATOL = 1e-5


@torch.inference_mode()
def predict_review(model, tokenizer, text, max_length, device):
    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    probabilities = model(**tokens).logits.softmax(dim=-1)[0].cpu().tolist()
    label = model.config.id2label[probabilities.index(max(probabilities))]
    return label, {model.config.id2label[index]: value for index, value in enumerate(probabilities)}


@torch.inference_mode()
def benchmark(model, tokenizer, texts, max_length, device):
    tokens = tokenizer(texts, truncation=True, max_length=max_length)
    inputs = [{key: value[index] for key, value in tokens.items()} for index in range(len(texts))]
    collator = DataCollatorWithPadding(tokenizer)
    results = []

    for batch_size in BATCH_SIZES:
        batches = [collator(inputs[start:start + batch_size]).to(device) for start in range(0, len(texts), batch_size)]
        for batch in batches:
            model(**batch)

        runs_ms = []
        for _ in range(REPEATS):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = perf_counter()
            for batch in batches:
                model(**batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runs_ms.append((perf_counter() - start) * 1000)
        results.append({"batch_size": batch_size, "runs_ms": runs_ms, "median_ms": median(runs_ms)})
    return results


def main():
    parser = argparse.ArgumentParser(description="Проверка загрузки и замер прямого прохода ruBERT")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--model", type=Path, default=Path(".artifacts/rubert_best"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    reference = json.loads((args.results / "rubert_reload_reference.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.data / "protocol.json").read_text(encoding="utf-8"))
    with (args.data / "validation.jsonl").open(encoding="utf-8") as file:
        reviews = {row["id"]: row["text"] for row in map(json.loads, file)}

    torch.set_num_threads(reference["num_threads"])
    device = torch.device(reference["device"])
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    review_id = protocol["reload_validation_id"]
    label, probabilities = predict_review(model, tokenizer, reviews[review_id], reference["max_length"], device)
    difference = max(abs(probabilities[name] - old) for name, old in reference["probabilities"].items())
    check = {
        "id": review_id,
        "device": str(device),
        "num_threads": torch.get_num_threads(),
        "reference_label": reference["prediction"],
        "reloaded_label": label,
        "reference_probabilities": reference["probabilities"],
        "reloaded_probabilities": probabilities,
        "max_abs_difference": difference,
        "atol": ATOL,
        "passed": review_id == reference["id"] and label == reference["prediction"] and difference <= ATOL,
    }
    (args.results / "rubert_reload_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    texts = [reviews[review_id] for review_id in protocol["latency_validation_ids"]]
    latency = {"device": str(device), "num_threads": torch.get_num_threads(), "n_reviews": len(texts), "results": benchmark(model, tokenizer, texts, reference["max_length"], device)}
    (args.results / "rubert_latency.json").write_text(json.dumps(latency, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
