"""Build retain99_merged_all_10_lang from full_merged_all_10_lang.

This keeps the finetune99 dataset on the same source/template as the
finetune100 dataset by removing the final forget01 block from each language.

Usage:
    python build_retain99.py

Run checklist:
    1. Run from the NLP_rice repository root.
    2. Ensure datasets/pyarrow are installed in the active Python environment.
    3. Ensure ./dataset/retain99_merged_all_10_lang does not already exist.
    4. Confirm the script reports 39,600 total rows.
    5. Confirm each language reports retained=3960 and removed=40.
    6. Finetune with: torchrun ... finetune.py --config-name finetune99
"""

from pathlib import Path

import datasets
from datasets import DatasetDict


LANGUAGES = ["en", "ar", "fa", "fr", "hi", "id", "iw", "ja", "ko", "ru"]
INPUT_DATASET = Path("./dataset/full_merged_all_10_lang")
OUTPUT_DATASET = Path("./dataset/retain99_merged_all_10_lang")
FULL_ROWS_PER_LANGUAGE = 4000
FORGET_ROWS_PER_LANGUAGE = 40
RETAIN_ROWS_PER_LANGUAGE = FULL_ROWS_PER_LANGUAGE - FORGET_ROWS_PER_LANGUAGE
EXPECTED_OUTPUT_ROWS = RETAIN_ROWS_PER_LANGUAGE * len(LANGUAGES)
OUTPUT_COLUMNS = ["question", "answer", "language"]


def row_key(row):
    return row["question"], row["answer"], row["language"]


def collect_language_indices(dataset):
    indices = {language: [] for language in LANGUAGES}
    for index, row in enumerate(dataset):
        language = row["language"]
        if language not in indices:
            raise ValueError(f"Unexpected language in input dataset: {language}")
        indices[language].append(index)

    bad_counts = {
        language: len(language_indices)
        for language, language_indices in indices.items()
        if len(language_indices) != FULL_ROWS_PER_LANGUAGE
    }
    if bad_counts:
        raise ValueError(f"Expected {FULL_ROWS_PER_LANGUAGE} rows per language, got {bad_counts}")

    return indices


def build_retain_dataset(full_dataset, language_indices):
    retained_parts = []
    removed_keys = set()
    retained_counts = {}
    removed_counts = {}

    for language in LANGUAGES:
        language_dataset = full_dataset.select(language_indices[language])
        if len(language_dataset) != FULL_ROWS_PER_LANGUAGE:
            raise ValueError(
                f"{language}: expected {FULL_ROWS_PER_LANGUAGE} rows, got {len(language_dataset)}"
            )

        retain_part = language_dataset.select(range(RETAIN_ROWS_PER_LANGUAGE))
        forget_part = language_dataset.select(range(RETAIN_ROWS_PER_LANGUAGE, FULL_ROWS_PER_LANGUAGE))

        retained_counts[language] = len(retain_part)
        removed_counts[language] = len(forget_part)
        removed_keys.update(row_key(row) for row in forget_part)
        retained_parts.append(retain_part)

    retain_dataset = datasets.concatenate_datasets(retained_parts)

    for column in list(retain_dataset.column_names):
        if column not in OUTPUT_COLUMNS:
            retain_dataset = retain_dataset.remove_columns(column)

    retain_dataset = retain_dataset.select_columns(OUTPUT_COLUMNS)

    retained_keys = {row_key(row) for row in retain_dataset}
    leaked_keys = retained_keys & removed_keys
    if leaked_keys:
        raise ValueError(f"Removed forget rows leaked into retain dataset: {len(leaked_keys)}")

    if len(retain_dataset) != EXPECTED_OUTPUT_ROWS:
        raise ValueError(f"Expected {EXPECTED_OUTPUT_ROWS} output rows, got {len(retain_dataset)}")

    for language, count in retained_counts.items():
        if count != RETAIN_ROWS_PER_LANGUAGE:
            raise ValueError(f"{language}: expected {RETAIN_ROWS_PER_LANGUAGE} retained rows, got {count}")

    for language, count in removed_counts.items():
        if count != FORGET_ROWS_PER_LANGUAGE:
            raise ValueError(f"{language}: expected {FORGET_ROWS_PER_LANGUAGE} removed rows, got {count}")

    if retain_dataset.column_names != OUTPUT_COLUMNS:
        raise ValueError(f"Unexpected output columns: {retain_dataset.column_names}")

    return retain_dataset, retained_counts, removed_counts


def main():
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Input dataset not found: {INPUT_DATASET}")

    if OUTPUT_DATASET.exists():
        raise FileExistsError(
            f"Output dataset already exists: {OUTPUT_DATASET}. "
            "Move or delete it before rebuilding."
        )

    full_dataset = datasets.load_from_disk(str(INPUT_DATASET))["train"]
    language_indices = collect_language_indices(full_dataset)

    retain_dataset, retained_counts, removed_counts = build_retain_dataset(full_dataset, language_indices)
    DatasetDict({"train": retain_dataset}).save_to_disk(str(OUTPUT_DATASET))

    print(f"Saved: {OUTPUT_DATASET}")
    print(f"Total rows: {len(retain_dataset)}")
    for language in LANGUAGES:
        print(
            f"{language}: retained={retained_counts[language]}, "
            f"removed={removed_counts[language]}"
        )


if __name__ == "__main__":
    main()
