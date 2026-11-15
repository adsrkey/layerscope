"""layerscope — разбор Qwen3.5/3.8 на MLX: от промпта к ответу и какие слои за него отвечают.

Команды (python -m layerscope <команда> --help):
    trace          подробный разбор одного промпта или пакета промптов, report.html
    layer-map      карта важности слоёв: отключение по одному и угловое расстояние
    patch          activation patching между двумя промптами
    tuned-lens     обучить tuned lens (точнее обычного logit lens)
    quant-compare  где квантизация искажает модель
    report         пересобрать report.html из готовых журналов, без модели
"""

__version__ = "0.3.0"
