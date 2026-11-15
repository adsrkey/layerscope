"""Точка входа: python -m qwen_trace <команда> ...  (или команда qwen-trace после pip install)."""
import importlib
import sys

COMMANDS = {
    "trace": ("cli", "подробный разбор одного промпта или пакета промптов, report.html"),
    "layer-map": ("layer_map", "карта важности слоёв"),
    "patch": ("patching", "activation patching между двумя промптами"),
    "tuned-lens": ("tuned_lens", "обучить tuned lens"),
    "quant-compare": ("quant_compare", "сравнение двух вариантов модели по слоям"),
    "report": ("report.cli", "пересобрать report.html из готовых журналов (модель не нужна)"),
}


def usage() -> str:
    width = max(map(len, COMMANDS))
    rows = "\n".join(f"  {name:<{width}}  {text}" for name, (_, text) in COMMANDS.items())
    return (f"использование: python -m qwen_trace <команда> [аргументы]\n\nкоманды:\n{rows}\n\n"
            f"Справка по команде: python -m qwen_trace <команда> --help.\n"
            f"Если первый аргумент — не команда, выполняется trace "
            f"(python -m qwen_trace МОДЕЛЬ \"промпт\").")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0
    if argv[0] in COMMANDS:
        module, args = COMMANDS[argv[0]][0], argv[1:]
    else:
        module, args = COMMANDS["trace"][0], argv
    return importlib.import_module(f"qwen_trace.{module}").main(args)


if __name__ == "__main__":
    sys.exit(main())
