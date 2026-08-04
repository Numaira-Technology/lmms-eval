import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lmms_eval.models.simple.qwen3_vl import Qwen3_VL, _is_video_path


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_id):
        return "<eos>"


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[10, 11]]))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ["prompt"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["final answer"]


class _FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12]])


class _VideoMetadata:
    def __init__(self, frames_indices):
        self.frames_indices = np.asarray(frames_indices)


class TestQwen3VLSimple(unittest.TestCase):
    def test_is_video_path_supports_common_video_extensions(self):
        for path in ("clip.mp4", "clip.avi", "clip.mov", "clip.mkv", "clip.webm", "clip.mpeg", "clip.mpg", "clip.MPEG"):
            with self.subTest(path=path):
                self.assertTrue(_is_video_path(path))

        self.assertFalse(_is_video_path("frame.jpg"))
        self.assertFalse(_is_video_path(None))

    def _make_model(self, max_num_frames=3):
        model = Qwen3_VL.__new__(Qwen3_VL)
        model._tokenizer = _FakeTokenizer()
        model.processor = _FakeProcessor()
        model._model = _FakeModel()
        model.max_pixels = 1024
        model.min_pixels = 256
        model.total_pixels = None
        model.max_num_frames = max_num_frames
        model.fps = None
        model.enable_thinking = None
        model.system_prompt = "You are a helpful assistant."
        model.interleave_visuals = False
        model.reasoning_prompt = None
        model.batch_size_per_gpu = 1
        model.use_cache = False
        model.device_map = "cpu"
        model._device = torch.device("cpu")
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo_task": {"test": [{"id": 0}]}}
        model.cache_hook = types.SimpleNamespace(add_partial=lambda *args, **kwargs: None)
        return model

    def test_generate_until_passes_video_metadata_and_kwargs_to_processor(self):
        model = self._make_model(max_num_frames=3)
        metadata = _VideoMetadata([0, 10, 20, 30, 40])
        video_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        captured_messages = []
        request = types.SimpleNamespace(
            args=("Describe the video", {}, lambda doc: ["demo.mp4"], 0, "demo_task", "test"),
        )

        def fake_process_vision_info(messages, **kwargs):
            captured_messages.append(messages)
            return None, [(video_tensor.clone(), metadata)], {"fps": 30.0, "max_frames": 5}

        with patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", side_effect=fake_process_vision_info):
            result = model.generate_until([request])

        self.assertEqual(result, ["final answer"])
        self.assertEqual(len(model.processor.calls), 1)

        processor_call = model.processor.calls[0]
        self.assertTrue(torch.equal(processor_call["videos"][0], video_tensor))
        self.assertIs(processor_call["video_metadata"][0], metadata)
        self.assertEqual(processor_call["fps"], 30.0)
        self.assertEqual(processor_call["max_frames"], 5)
        self.assertTrue(np.array_equal(metadata.frames_indices, np.array([0, 10, 20, 30, 40])))

        video_content = captured_messages[0][0][1]["content"][0]
        self.assertEqual(video_content["video"], "demo.mp4")
        self.assertEqual(video_content["nframes"], 3)
        self.assertEqual(video_content["min_pixels"], 256)
        self.assertEqual(video_content["max_pixels"], 1024)

    def test_build_video_kwargs_uses_qwen_vl_utils_frame_contract(self):
        model = self._make_model(max_num_frames=3)
        self.assertEqual(model._build_video_kwargs(), {"min_pixels": 256, "nframes": 3, "max_pixels": 1024})

        model.fps = 2.0
        self.assertEqual(model._build_video_kwargs(), {"min_pixels": 256, "fps": 2.0, "max_frames": 3, "max_pixels": 1024})

        model.fps = None
        model.total_pixels = 4096
        self.assertEqual(model._build_video_kwargs(), {"min_pixels": 256, "max_frames": 3, "total_pixels": 4096})


if __name__ == "__main__":
    unittest.main()
