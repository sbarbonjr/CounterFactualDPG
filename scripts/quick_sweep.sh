#!/bin/bash
# Quick Sweep Helper Script
# Makes it easy to initialize and run WandB sweeps for DPG hyperparameter optimization
#
# Usage:
#   ./scripts/quick_sweep.sh init <metric>     - Initialize a new sweep
#   ./scripts/quick_sweep.sh run <sweep_id>    - Run an agent for a sweep
#   ./scripts/quick_sweep.sh list              - List available metrics

ENTITY="mllab-ts-universit-di-trieste"
PROJECT="CounterFactualDPG"
DATASET="iris"
PYTHON=".venv/bin/python"

case "$1" in
    init)
        METRIC="${2:-plausibility_sum}"
        echo "Initializing sweep for metric: $METRIC"
        $PYTHON scripts/run_sweep.py --init-sweep \
            --dataset "$DATASET" \
            --target-metric "$METRIC" \
            --entity "$ENTITY" \
            --project "$PROJECT"
        ;;
    
    run)
        if [ -z "$2" ]; then
            echo "ERROR: Sweep ID required"
            echo "Usage: $0 run <sweep_id>"
            exit 1
        fi
        SWEEP_ID="$2"
        COUNT="${3:-10}"
        echo "Running sweep agent (max $COUNT runs)"
        $PYTHON scripts/run_sweep.py --run-agent \
            --sweep-id "$SWEEP_ID" \
            --count "$COUNT" \
            --dataset "$DATASET" \
            --target-metric "plausibility_sum" \
            --entity "$ENTITY" \
            --project "$PROJECT"
        ;;
    
    list)
        $PYTHON scripts/run_sweep.py --list-metrics
        ;;
    
    *)
        echo "Usage:"
        echo "  $0 init [metric]        - Initialize sweep (default: plausibility_sum)"
        echo "  $0 run <sweep_id> [n]   - Run agent for sweep (default: 10 runs)"
        echo "  $0 list                 - List available metrics"
        echo ""
        echo "Examples:"
        echo "  $0 init plausibility_sum"
        echo "  $0 init perc_valid_cf"
        echo "  $0 run bmjavbdu"
        echo "  $0 run bmjavbdu 20"
        exit 1
        ;;
esac
