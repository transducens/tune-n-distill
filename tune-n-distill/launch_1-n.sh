#!/bin/bash
set -euo pipefail

bash launch-train.sh \
    --lang-1="en en_XX" \
    --lang-2="es es_XX" \
    --experiment-dir="en-es" \
    --mBART50-dir="/models" \
    --corpora-for-bicleaner-train="train" \
    --corpora-for-bicleaner-dev="dev" \
    --parallel-corpora="train" \
    --monolingual-corpora-lang-1="mono" \
    --monolingual-corpora-lang-2="mono" \
    --dev-corpus="dev" \
    --test-corpus="test"