from eduread.training import run_v3_training
from eduread.data import EduReadProcessedDatasetV3, validate_processed_index


def test_v3_index_validation_accepts_a_valid_entry():
    records = [{
        "example_id": "T001",
        "task_type": "mathematical_transcription",
        "image_filename": "example.png",
        "processed_path": "/private/example.pt",
        "supervised_tokens": 12,
    }]
    assert validate_processed_index(records, expected_count=1) == []


def test_v3_index_validation_reports_wrong_count():
    assert validate_processed_index([], expected_count=160)


def test_training_is_disabled_by_default_without_touching_model():
    plan = run_v3_training(
        model=None,
        processor=None,
        train_loader=None,
        eval_loader=None,
        optimizer=None,
        scheduler=None,
        group_token_counts=[],
        output_dir="this_path_must_not_be_created_by_a_dry_run",
    )
    assert plan["executed"] is False
    assert plan["planned_epochs"] == 3
    assert plan["planned_optimizer_updates"] == 120
