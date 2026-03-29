#!/usr/bin/env bash
set -euo pipefail

lab="${1:-}"

if [[ -z "${lab}" ]]; then
  echo "usage: $0 <lr1|lr2|lr3>" >&2
  exit 2
fi

case "${lab}" in
  lr1)
    ruff check --extend-select I --ignore E741 ../lr1 ../lr1/tests
    mypy --follow-imports=skip \
      ../lr1/application/analysis.py \
      ../lr1/application/services.py \
      ../lr1/application/viewmodels.py \
      ../lr1/domain/functions.py \
      ../lr1/domain/models.py \
      ../lr1/domain/numerical.py \
      ../lr1/domain/search.py \
      ../lr1/infrastructure/logging.py \
      ../lr1/infrastructure/settings.py
    bandit -q -r ../lr1 -x ../lr1/.tox,../lr1/tests
    pip-audit -r ../requirements.txt
    deptry ../lr1 ../lr1/tests --ignore DEP001,DEP002,DEP003 --requirements-files ../requirements.txt
    vulture ../lr1/application ../lr1/domain ../lr1/ui ../lr1/main.py ../lr1/__main__.py --min-confidence 80
    radon cc -s -n B ../lr1/application ../lr1/domain ../lr1/ui ../lr1/main.py ../lr1/__main__.py
    pytest -q tests/test_lr1_architecture.py
    ;;
  lr2)
    ruff check ../lr2 ../lr2/tests
    mypy ../lr2
    bandit -q -r ../lr2 -x ../lr2/.tox,../lr2/tests
    pip-audit -r ../requirements.txt
    deptry ../lr2 ../lr2/tests
    vulture ../lr2/application ../lr2/domain ../lr2/ui ../lr2/main.py ../lr2/__main__.py --min-confidence 80
    radon cc -s -n B ../lr2/application ../lr2/domain ../lr2/ui ../lr2/main.py ../lr2/__main__.py
    pytest -q tests/test_lr2_architecture.py
    ;;
  lr3)
    ruff check --extend-select I ../lr3 ../lr3/tests
    mypy ../lr3
    bandit -q -r ../lr3 -x ../lr3/.tox,../lr3/tests
    pip-audit -r ../requirements.txt
    deptry ../lr3 ../lr3/tests --ignore DEP001,DEP002,DEP003 --requirements-files ../requirements.txt
    vulture ../lr3/application ../lr3/domain ../lr3/ui ../lr3/main.py ../lr3/__main__.py --min-confidence 80
    radon cc -s -n B ../lr3/application ../lr3/domain ../lr3/ui ../lr3/main.py ../lr3/__main__.py
    pytest -q tests/test_lr3_architecture.py
    ;;
  *)
    echo "unknown lab '${lab}', expected lr1|lr2|lr3" >&2
    exit 2
    ;;
esac
