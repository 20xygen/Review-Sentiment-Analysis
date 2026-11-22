# Классификация отзывов RuReviews

Сравнение двух вариантов TF-IDF с логистической регрессией, дообученного `cointegrated/rubert-tiny2` и `Qwen3.5-4B`. LLM я брал через API сервиса Mixlayer, там бесплатно. Исходные данные и фиксированные выборки описаны в [Источнике данных](Источник%20данных.md)

## Локальный запуск

Запускайте команды из корня проекта. Нужен Python 3.13 с модулем `venv`

Windows (PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.7.1'
.\.venv\Scripts\python.exe -m pip install -r requirements.txt notebook
.\.venv\Scripts\python.exe -m notebook
```

Linux (bash)

```bash
python3.13 -m venv .venv
./.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch>=2.7.1'
./.venv/bin/python -m pip install -r requirements.txt notebook
./.venv/bin/python -m notebook
```

Создайте файл `.env` рядом с ноутбуком и запишите в него одну строку `MIXLAYER_API_KEY=...` (код расчитан на использование этого API и не тестировался с другими сервисами)

Откройте [Домашняя_NLP.ipynb](Домашняя_NLP.ipynb) и выполните ячейки по порядку. Подготовка данных скачивает зафиксированную версию RuReviews и проверяет контрольные суммы. ruBERT обучается на CPU. Ноутбук сохраняет лучший чекпойнт с токенизатором в `.artifacts/rubert_best/` и использует его при повторном запуске. Для повторного обучения с нуля сохраните старые логи отдельно и запустите ноутбук с пустой `.artifacts/`

Mixlayer выполняет 60 генераций для выбора промпта, 150 для теста и 12 для проверки устойчивости. Успешные ответы добавляются в JSONL сразу. При повторном запуске ячейка читает уже сохранённые ответы для той же версии запроса. Для нового прогона с изменённым промптом используйте пустые файлы результатов

## Результаты

Я делал и запускал работу на Windows. Сохранённые результаты получены на AMD Ryzen 7 4800H с Python 3.13.17 и PyTorch 2.15.0+cpu

- Выбор на validation до оценки на test записан в `results/selection.json`
- Итоговые метрики и матрицы ошибок сохранены в `results/metrics.jsonl` и `results/confusion_matrices.jsonl`
- Предсказания с ID находятся в `results/*predictions.jsonl`, результаты подбора порога в `results/review_threshold_validation.jsonl`
- Исходные ответы Mixlayer сохранены в `results/generator_validation.jsonl`, `results/generator_test.jsonl` и `results/generator_injection.jsonl`
- Логи и результаты проверки ruBERT находятся в `results/tensorboard/`, `results/rubert_reload_check.json` и `results/rubert_latency.json`

Для запуска TensorBoard используйте `.\.venv\Scripts\tensorboard.exe --logdir results/tensorboard` на Windows или `./.venv/bin/tensorboard --logdir results/tensorboard` на Linux. Вкладки Scalars и Text показывают метрики обучения и пример ошибки

После раздела 3 перезагрузку и замеры можно запустить отдельной командой `.\.venv\Scripts\python.exe reload_rubert.py` на Windows или `./.venv/bin/python reload_rubert.py` на Linux. Ноутбук запускает этот же скрипт в новом процессе

Контрольные вопросы в `answers.md`

## Использование ИИ

ИИ был использован лишь как вспомогательный инструмент. Я просил писать несущественный код (например, для Mixlayer) и сформировать некоторые части этого README. Ответы на контрольные вопросы написал полностью сам