from lmms_eval.models import MODEL_REGISTRY_V2, list_available_models

EXPECTED_MODELS = {
    "cambrians",
    "gemini",
    "openai",
    "qwen2_5_omni",
    "qwen2_5_vl",
    "qwen2_audio",
    "qwen2_vl",
    "qwen3_5",
    "qwen3_omni",
    "qwen3_vl",
    "qwen_image_edit",
    "qwen_vl",
    "qwen_vl_api",
}


def test_only_supported_model_families_are_registered():
    assert set(list_available_models()) == EXPECTED_MODELS


def test_molmo_and_gpt_use_openai_compatible_backend():
    assert MODEL_REGISTRY_V2.resolve("molmo").model_id == "openai"
    assert MODEL_REGISTRY_V2.resolve("gpt").model_id == "openai"
