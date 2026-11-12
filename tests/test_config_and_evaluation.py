from eduread import EduReadConfig
from eduread.evaluation import compare_prediction, normalize_text, summarize_predictions


def test_audited_v3_configuration():
    config = EduReadConfig()
    config.validate()
    assert config.model.model_id == "google/paligemma-3b-pt-448"
    assert config.training.total_optimizer_updates == 120
    assert config.dataset.train_records == 160
    assert config.dataset.evaluation_records == 40


def test_text_normalization():
    assert normalize_text("Hello,  WORLD!") == "hello world"


def test_exact_match_and_character_similarity():
    result = compare_prediction("A + B", "a b")
    assert result["exact_match"] is True
    assert result["character_similarity"] == 1.0


def test_summary_empty_input():
    assert summarize_predictions([]) == {
        "examples": 0,
        "exact_matches": 0,
        "mean_character_similarity": 0.0,
    }
