# Готовые инструменты рядом

Что нашлось на момент проверки (ноябрь 2026). Ни один не покрывает связку «MLX + `mlx-vlm` + гибрид `qwen3_5` + 4-битные веса», поэтому проект остаётся собственным; но часть шагов полезно сверять с готовыми инструментами.

| Шаг | Инструмент | Заметки |
|---|---|---|
| Хуки, logit lens, патчинг на MLX | [mlux](https://github.com/batson/mlux) | `mlx-lm`-совместимые модели, браузерные просмотрщики; про `mlx-vlm` и гибриды в README ничего нет |
| То же на MLX | [mlxterp](https://coairesearch.github.io/mlxterp) | по духу близок к nnsight; про `qwen3_5` и `mlx-vlm` не сказано |
| Хуки, разложение логита, патчинг (PyTorch) | [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) | экспериментальная поддержка гибридов Gated DeltaNet (Qwen3.5, Qwen3-Next), но нужны полноточные веса Hugging Face, а не ваша MLX-сборка; недавно чинили связанную ошибку ([PR #1803](https://github.com/TransformerLensOrg/TransformerLens/pull/1803)) |
| Tuned lens | [tuned-lens](https://tuned-lens.readthedocs.io/) | список поддерживаемых моделей не найден; код упоминает только старые семейства |
| Разреженные автоэнкодеры | [Qwen-Scope](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50) | для **Qwen3.5-27B**: все 64 слоя, остаточный поток, 81 920 признаков, 50 активных, формат PyTorch `.pt`; для Qwen3.8 применимость не подтверждена |
| Сравнение квантизаций (GGUF) | [llama-perplexity](https://www.mintlify.com/ggml-org/llama.cpp/api/tools/llama-perplexity) | `--kl-divergence` с базовым файлом логитов; только GGUF, послойной ошибки нет |
| Выбор слоёв на удаление | [Gromov et al.](https://arxiv.org/pdf/2403.17887), [ShortGPT](https://huggingface.co/deepnet/ShortGpt) | метод, на котором основано угловое расстояние; в неофициальном репозитории оно помечено как незавершённое |

Captum и Inseq (градиентные методы вклада токенов) и поддержка nnsight не проверялись.
