"""Запуск всех тестов без pytest:  python tests/run_all.py [часть имени файла или теста]

Тесты идут на крошечной случайной модели той же архитектуры (tests/tiny.py), веса скачивать
не нужно. Если установлен pytest, можно и так:  pytest tests
"""
import importlib
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def main(argv):
    want = argv[0] if argv else ""
    failed = ran = 0
    for path in sorted(HERE.glob("test_*.py")):
        mod = importlib.import_module(path.stem)
        for name, fn in list(vars(mod).items()):
            if not (name.startswith("test_") and callable(fn)) or want not in f"{path.stem}.{name}":
                continue
            ran += 1
            try:
                fn()
                print(f"ok    {path.stem}.{name}")
            except Exception:
                failed += 1
                print(f"FAIL  {path.stem}.{name}")
                traceback.print_exc()
    print(f"\n{'все тесты пройдены' if not failed else f'упало: {failed}'} (запущено {ran})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
