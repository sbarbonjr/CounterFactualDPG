#!/bin/bash
# Setup script for the experiment tracking system

set -e

echo "=================================="
echo "CounterFactual DPG Setup"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
echo ""

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Check WandB installation
echo "Checking WandB installation..."
python3 -c "import wandb; print(f'✓ WandB version: {wandb.__version__}')"
echo ""

# Check if logged in to WandB
echo "Checking WandB authentication..."
if wandb verify 2>/dev/null; then
    echo "✓ Already logged in to WandB"
else
    echo "⚠ Not logged in to WandB"
    echo ""
    echo "Please run: wandb login"
    echo "Get your API key from: https://wandb.ai/authorize"
fi
echo ""

# Test config loading
echo "Testing configuration system..."
python3 -c "
import yaml
with open('configs/experiment_config.yaml') as f:
    config = yaml.safe_load(f)
print(f'✓ Config loaded: {config[\"experiment\"][\"name\"]}')
"
echo ""

# Create test output directory
echo "Setting up output directories..."
mkdir -p experiment_results
echo "✓ Output directories ready"
echo ""

echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Login to WandB (if not already): wandb login"
echo "  2. Run a quick test: python3 scripts/run_experiment.py --config configs/quick_test.yaml"
echo "  3. View results: wandb dashboard or python3 scripts/query_results.py list"
echo ""
