#!/bin/bash

################################################################################
# Logging Run Script for AgentSR
#
# This script runs the symbolic regression agent on datasets with reasoning
# logging enabled (--log-reasoning flag).
#
# Usage:
#   ./run_with_logging.sh <category> [options]
#
# Categories:
#   bio_pop_growth  - Biological population growth equations
#   chem_react      - Chemical reaction equations
#   matsci          - Materials science equations
#   phys_osc        - Physics oscillation equations
#   lsr_transform   - Transformed Feynman equations (large set)
#
# Options:
#   --agent-model MODEL  Model for SR agent (default: gpt-4o-mini)
#   --temperature TEMP   Temperature for agent LLM (default: 0.7)
#   --max-iter N         Maximum iterations (default: 5)
#   --limit N            Limit to first N datasets (for testing)
#   --help               Show this help message
#
# Examples:
#   ./run_with_logging.sh bio_pop_growth
#   ./run_with_logging.sh lsr_transform --limit 10
#   ./run_with_logging.sh phys_osc --agent-model gpt-4o
################################################################################

set -e  # Exit on error

# Default values
AGENT_MODEL="gpt-4o-mini"
TEMPERATURE="0.7"
MAX_ITER="5"
LIMIT=""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/llmsr/data/llmsrbench/csv"
AGENTSR_DIR="$SCRIPT_DIR/src"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show help
show_help() {
    cat << EOF
Logging Run Script for AgentSR

Usage: $0 <category> [options]

Categories:
  bio_pop_growth  - Biological population growth equations
  chem_react      - Chemical reaction equations
  matsci          - Materials science equations
  phys_osc        - Physics oscillation equations
  lsr_transform   - Transformed Feynman equations (large set)

Options:
  --agent-model MODEL  Model for SR agent (default: gpt-4o-mini)
  --temperature TEMP   Temperature for agent LLM (default: 0.7)
  --max-iter N         Maximum iterations (default: 5)
  --limit N            Limit to first N datasets (for testing)
  --help               Show this help message

Examples:
  $0 bio_pop_growth
  $0 lsr_transform --limit 10
  $0 phys_osc --agent-model gpt-4o
EOF
    exit 0
}

# Parse arguments
if [ $# -eq 0 ]; then
    print_error "No category specified"
    show_help
fi

CATEGORY="$1"
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --agent-model)
            AGENT_MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-iter)
            MAX_ITER="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Validate category and set data path
case "$CATEGORY" in
    bio_pop_growth|chem_react|matsci|phys_osc)
        DATA_PATH="$DATA_DIR/lsr_synth/$CATEGORY"
        ;;
    lsr_transform)
        DATA_PATH="$DATA_DIR/lsr_transform"
        ;;
    *)
        print_error "Invalid category: $CATEGORY"
        echo "Valid categories: bio_pop_growth, chem_react, matsci, phys_osc, lsr_transform"
        exit 1
        ;;
esac

# Check if data directory exists
if [ ! -d "$DATA_PATH" ]; then
    print_error "Data directory not found: $DATA_PATH"
    exit 1
fi

# Print configuration
echo ""
print_info "========================================"
print_info "AgentSR Logging Run"
print_info "========================================"
print_info "Category:        $CATEGORY"
print_info "Data Path:       $DATA_PATH"
print_info "Agent Model:     $AGENT_MODEL"
print_info "Temperature:     $TEMPERATURE"
print_info "Max Iterations:  $MAX_ITER"
[ -n "$LIMIT" ] && print_info "Limit:           $LIMIT datasets"
print_info "========================================"
echo ""

# Extract unique dataset names (remove _train.csv and _test.csv suffixes)
print_info "Discovering datasets..."
DATASETS=($(ls "$DATA_PATH"/*.csv 2>/dev/null | \
    sed 's/_train\.csv$//' | \
    sed 's/_test\.csv$//' | \
    sed 's/_ood_test\.csv$//' | \
    sort -u | \
    xargs -n1 basename))

if [ ${#DATASETS[@]} -eq 0 ]; then
    print_error "No datasets found in $DATA_PATH"
    exit 1
fi

print_success "Found ${#DATASETS[@]} datasets"

# Apply limit if specified
if [ -n "$LIMIT" ]; then
    DATASETS=("${DATASETS[@]:0:$LIMIT}")
    print_warning "Limited to first $LIMIT datasets"
fi

# Change to agentsr/src directory
cd "$AGENTSR_DIR"

# Initialize counters
TOTAL=${#DATASETS[@]}
SUCCESSFUL=0
FAILED=0

echo ""
echo "==================================================="
echo "Starting runs at $(date)"
echo "==================================================="
echo ""

# Process each dataset sequentially
current_num=0
for dataset in "${DATASETS[@]}"; do
    current_num=$((current_num + 1))

    echo "==================================================="
    echo "[$current_num/$TOTAL] Processing: $dataset"
    echo "Started at: $(date)"
    echo "==================================================="

    if python main.py \
        -D "$dataset" \
        --log-reasoning \
        --model "$AGENT_MODEL" \
        --temperature "$TEMPERATURE" \
        --max_iter "$MAX_ITER"; then

        SUCCESSFUL=$((SUCCESSFUL + 1))
        print_success "[$current_num/$TOTAL] Completed: $dataset"
    else
        FAILED=$((FAILED + 1))
        print_error "[$current_num/$TOTAL] Failed: $dataset"
    fi

    echo ""
done

# Print final summary
echo ""
echo "==================================================="
echo "Run Complete at $(date)"
echo "==================================================="
print_info "Category:        $CATEGORY"
print_info "Total Datasets:  $TOTAL"
print_success "Successful:      $SUCCESSFUL"
print_error "Failed:          $FAILED"
print_info "Reasoning logs saved to: $AGENTSR_DIR/../logs/"
echo "==================================================="
echo ""

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
