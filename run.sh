#!/bin/sh
set -eu

export PYTHONUNBUFFERED=1

MODEL_NAME="${MODEL_NAME:-rhea-only-assist-max}"
TASKS="${TASKS:-refinement_multi expansion_multi follow-up_multi recollection_multi_cls recollection_multi_global-inst}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
RHEA_MODE="${RHEA_MODE:-2}"

for TASK_NAME in $TASKS; do
    echo "======================================================================="
    echo "Running inference: model=${MODEL_NAME} task=${TASK_NAME}"
    echo "======================================================================="
    python inference.py "$MODEL_NAME" "$TASK_NAME" \
        --rhea \
        --w_retieval_2 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --mode "$RHEA_MODE"
done

python evaluate.py "$MODEL_NAME"
python calculate_score.py

echo "============================== All jobs completed =============================="
