import hashlib
from http.client import RemoteDisconnected
import json
from pathlib import Path
from time import sleep
from urllib.request import Request
from urllib.request import urlopen


URL = "https://mixlayer.ai/v1/chat/completions"
MODEL = "qwen/qwen3.5-4b-free"
SETTINGS = {"temperature": 0.0, "max_tokens": 64, "reasoning_effort": "none"}
RETRY_DELAY = 2
TIMEOUT = 90
LABELS = ("negative", "neutral", "positive")
INVALID = "__invalid__"


def get_api_key():
    return Path(".env").read_text(encoding="utf-8-sig").split("=", 1)[1].strip()


def request_completion(body, api_key):
    request = Request(URL, data=body, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    try:
        response = urlopen(request, timeout=TIMEOUT)
    except RemoteDisconnected:
        sleep(RETRY_DELAY)
        response = urlopen(request, timeout=TIMEOUT)
    with response:
        return json.load(response)["choices"][0]["message"]["content"]


def parse_label(raw):
    if not isinstance(raw, str):
        return INVALID
    try:
        pairs = json.loads(raw, object_pairs_hook=tuple)
    except json.JSONDecodeError:
        return INVALID
    if isinstance(pairs, tuple) and len(pairs) == 1:
        key, value = pairs[0]
        if key == "label" and value in LABELS:
            return value
    return INVALID


def generate_responses(cases, output):
    cached = {}
    if output.exists():
        with output.open(encoding="utf-8") as file:
            for line in file:
                row = json.loads(line)
                cached[row["case_id"]] = row

    rows = []
    api_key = None
    with output.open("a", encoding="utf-8") as file:
        for case in cases:
            metadata = {key: value for key, value in case.items() if key != "messages"}
            payload = {"model": MODEL, "messages": case["messages"], **SETTINGS}
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            fingerprint = hashlib.sha256(body).hexdigest()
            row = cached.get(case["case_id"])
            if row is not None and row["prompt_sha256"] != fingerprint:
                raise ValueError(f"Настройки запроса изменились. Сохраните старые ответы из {output} отдельно.")
            if row is None:
                if api_key is None:
                    api_key = get_api_key()
                raw = request_completion(body, api_key)
                row = {**metadata, "model": MODEL, "prompt_sha256": fingerprint, "raw_response": raw, "prediction": parse_label(raw)}
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
            rows.append(row)
    return rows
