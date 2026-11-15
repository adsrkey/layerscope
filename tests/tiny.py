"""Крошечная случайная модель qwen3_5 и игрушечный токенизатор — для тестов без скачивания весов."""
from types import SimpleNamespace

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

VOCAB = 96


class ToyTokenizer:
    """Один символ ASCII = один токен; 0 — конец ответа."""
    eos_token_id = 0

    def encode(self, text):
        return [min(max(ord(c) - 31, 1), VOCAB - 1) for c in text]

    def decode(self, ids):
        return "".join(chr(i + 31) for i in ids if i)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=False, **kw):
        return "".join(f"<{m['role'][0]}>{m['content']}" for m in messages) + "<a>"


def tiny_model(seed: int = 0, layers: int = 8, hidden: int = 64, bits: int = 0, gain: float = 3.0):
    """bits > 0 — те же веса, но квантизованные (как в MLX-сборках моделей)."""
    from mlx_vlm.models.qwen3_5.config import TextConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel
    mx.random.seed(seed)
    args = TextConfig(
        model_type="qwen3_5", hidden_size=hidden, intermediate_size=hidden * 2,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32,
        linear_value_head_dim=32, linear_conv_kernel_dim=4, num_hidden_layers=layers,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=VOCAB, num_key_value_heads=2,
        max_position_embeddings=512, eos_token_id=0, tie_word_embeddings=False, head_dim=8,
        rope_parameters={"type": "default", "mrope_section": [1, 1, 0], "rope_theta": 10000,
                         "partial_rotary_factor": 0.25})
    cfg = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2), image_token_id=VOCAB + 1,
                          video_token_id=VOCAB + 2, vision_start_token_id=VOCAB + 3,
                          vision_end_token_id=VOCAB + 4, eos_token_id=0)
    lm = LanguageModel(args, cfg)
    # случайные веса покрупнее, иначе все слои почти ничего не меняют
    params = [(k, v if v.ndim < 2 else v * gain) for k, v in tree_flatten(lm.parameters())]
    lm.update(tree_unflatten(params))
    if bits:
        import mlx.nn as nn
        nn.quantize(lm, group_size=32, bits=bits)
    mx.eval(lm.parameters())
    lm.eval()
    model = SimpleNamespace(language_model=lm, config=cfg)
    return model, ToyTokenizer()
