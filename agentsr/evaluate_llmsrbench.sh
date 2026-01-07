#!/bin/bash

################################################################################
# Benchmark Evaluation Script for AgentSR
#
# This script evaluates the symbolic regression agent on LLM-SRBench datasets
# with support for parallel execution (up to 12 concurrent jobs).
#
# Usage:
#   ./evaluate_benchmark.sh <category> [options]
#
# Categories:
#   bio_pop_growth  - Biological population growth equations
#   chem_react      - Chemical reaction equations
#   matsci          - Materials science equations
#   phys_osc        - Physics oscillation equations
#   lsr_transform   - Transformed Feynman equations (large set)
#
# Options:
#   --skip-symbolic      Skip symbolic accuracy evaluation (faster)
#   --eval-model MODEL   Model for symbolic evaluation (default: gpt-4o)
#   --agent-model MODEL  Model for SR agent (default: gpt-4o-mini)
#   --temperature TEMP   Temperature for agent LLM (default: 0.7)
#   --max-iter N         Maximum iterations (default: 5)
#   --parallel N         Number of parallel jobs (1-12, default: 1)
#   --resume             Resume from existing results (skip completed datasets)
#   --limit N            Limit to first N datasets (for testing)
#   --help               Show this help message
#
# Examples:
#   ./evaluate_benchmark.sh bio_pop_growth
#   ./evaluate_benchmark.sh lsr_transform --skip-symbolic --limit 10
#   ./evaluate_benchmark.sh phys_osc --agent-model gpt-4o --resume
#   ./evaluate_benchmark.sh lsr_transform --parallel 12 --skip-symbolic
################################################################################

set -e  # Exit on error

# Default values
SKIP_SYMBOLIC=""
EVAL_MODEL="gpt-4o"
AGENT_MODEL="gpt-4o-mini"
TEMPERATURE="0.7"
MAX_ITER="5"
PARALLEL=1
RESUME=false
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
Benchmark Evaluation Script for AgentSR (with Parallel Execution)

Usage: $0 <category> [options]

Categories:
  bio_pop_growth  - Biological population growth equations
  chem_react      - Chemical reaction equations
  matsci          - Materials science equations
  phys_osc        - Physics oscillation equations
  lsr_transform   - Transformed Feynman equations (large set)

Options:
  --skip-symbolic      Skip symbolic accuracy evaluation (faster)
  --eval-model MODEL   Model for symbolic evaluation (default: gpt-4o)
  --agent-model MODEL  Model for SR agent (default: gpt-4o-mini)
  --temperature TEMP   Temperature for agent LLM (default: 0.7)
  --max-iter N         Maximum iterations (default: 5)
  --parallel N         Number of parallel jobs (1-12, default: 1)
  --resume             Resume from existing results (skip completed datasets)
  --limit N            Limit to first N datasets (for testing)
  --help               Show this help message

Examples:
  $0 bio_pop_growth
  $0 lsr_transform --skip-symbolic --limit 10
  $0 phys_osc --agent-model gpt-4o --resume
  $0 lsr_transform --parallel 12 --skip-symbolic
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
        --skip-symbolic)
            SKIP_SYMBOLIC="--skip-symbolic"
            shift
            ;;
        --eval-model)
            EVAL_MODEL="$2"
            shift 2
            ;;
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
        --parallel)
            PARALLEL="$2"
            if [ "$PARALLEL" -lt 1 ] || [ "$PARALLEL" -gt 12 ]; then
                print_error "Parallel jobs must be between 1 and 12"
                exit 1
            fi
            shift 2
            ;;
        --resume)
            RESUME=true
            shift
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

# Set output directory
OUTPUT_DIR="$SCRIPT_DIR/evaluation_results/$CATEGORY"
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo ""
print_info "========================================"
print_info "AgentSR Benchmark Evaluation"
print_info "========================================"
print_info "Category:        $CATEGORY"
print_info "Data Path:       $DATA_PATH"
print_info "Output Dir:      $OUTPUT_DIR"
print_info "Agent Model:     $AGENT_MODEL"
print_info "Eval Model:      $EVAL_MODEL"
print_info "Temperature:     $TEMPERATURE"
print_info "Max Iterations:  $MAX_ITER"
print_info "Parallel Jobs:   $PARALLEL"
print_info "Skip Symbolic:   $([ -n "$SKIP_SYMBOLIC" ] && echo "Yes" || echo "No")"
print_info "Resume:          $([ "$RESUME" = true ] && echo "Yes" || echo "No")"
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

# Check for existing results if resuming
COMPLETED_DATASETS=()
if [ "$RESUME" = true ] && [ -d "$OUTPUT_DIR" ]; then
    for dataset in "${DATASETS[@]}"; do
        if [ -f "$OUTPUT_DIR/${dataset}_eval.json" ]; then
            COMPLETED_DATASETS+=("$dataset")
        fi
    done
    if [ ${#COMPLETED_DATASETS[@]} -gt 0 ]; then
        print_info "Found ${#COMPLETED_DATASETS[@]} completed datasets (will skip)"
    fi
fi

# Change to agentsr/src directory
cd "$AGENTSR_DIR"

# Initialize counters
TOTAL=${#DATASETS[@]}
SUCCESSFUL=0
FAILED=0
SKIPPED=0

# Create a log file
LOG_FILE="$OUTPUT_DIR/evaluation_log_$(date +%Y%m%d_%H%M%S).txt"
print_info "Logging to: $LOG_FILE"
echo ""

# Create a results tracking file
RESULTS_FILE="$OUTPUT_DIR/.evaluation_results_$(date +%Y%m%d_%H%M%S).txt"
touch "$RESULTS_FILE"

echo "==================================================="
echo "Starting evaluation at $(date)"
echo "==================================================="
echo ""

# Function to evaluate a single dataset
evaluate_dataset() {
    local dataset="$1"
    local dataset_num="$2"
    local total="$3"
    local log_file="$OUTPUT_DIR/${dataset}_log.txt"

    {
        echo "==================================================="
        echo "[$dataset_num/$total] Evaluating: $dataset"
        echo "Started at: $(date)"
        echo "==================================================="

        if python main.py \
            -D "$dataset" \
            --eval \
            --model "$AGENT_MODEL" \
            --eval-model "$EVAL_MODEL" \
            --temperature "$TEMPERATURE" \
            --max_iter "$MAX_ITER" \
            --output-dir "$OUTPUT_DIR" \
            $SKIP_SYMBOLIC 2>&1; then

            echo "SUCCESS:$dataset" >> "$RESULTS_FILE"
            echo ""
            echo "[SUCCESS] [$dataset_num/$total] Completed: $dataset at $(date)"
        else
            echo "FAILED:$dataset" >> "$RESULTS_FILE"
            echo ""
            echo "[ERROR] [$dataset_num/$total] Failed: $dataset at $(date)"
        fi

        echo "==================================================="
    } >> "$log_file" 2>&1
}

# Export variables and functions for parallel execution
export AGENT_MODEL EVAL_MODEL TEMPERATURE MAX_ITER OUTPUT_DIR SKIP_SYMBOLIC RESULTS_FILE
export -f evaluate_dataset

# Prepare dataset list (skip already completed if resuming)
DATASETS_TO_PROCESS=()
DATASET_NUMS=()
current_num=0

for dataset in "${DATASETS[@]}"; do
    current_num=$((current_num + 1))

    # Skip if already completed and resuming
    if [ "$RESUME" = true ]; then
        if [[ " ${COMPLETED_DATASETS[@]} " =~ " ${dataset} " ]]; then
            print_warning "[$current_num/$TOTAL] Skipping $dataset (already completed)"
            SKIPPED=$((SKIPPED + 1))
            echo "SKIPPED:$dataset" >> "$RESULTS_FILE"
            continue
        fi
    fi

    DATASETS_TO_PROCESS+=("$dataset")
    DATASET_NUMS+=("$current_num")
done

# Run evaluations in parallel
if [ ${#DATASETS_TO_PROCESS[@]} -gt 0 ]; then
    print_info "Processing ${#DATASETS_TO_PROCESS[@]} datasets with $PARALLEL parallel jobs..."
    echo ""

    # Use GNU parallel if available, otherwise fall back to xargs
    if command -v parallel &> /dev/null; then
        print_info "Using GNU parallel for execution"
        for i in "${!DATASETS_TO_PROCESS[@]}"; do
            echo "${DATASETS_TO_PROCESS[$i]} ${DATASET_NUMS[$i]} $TOTAL"
        done | parallel --jobs "$PARALLEL" --colsep ' ' evaluate_dataset {1} {2} {3}
    else
        print_info "GNU parallel not found, using xargs (less efficient)"
        for i in "${!DATASETS_TO_PROCESS[@]}"; do
            echo "${DATASETS_TO_PROCESS[$i]} ${DATASET_NUMS[$i]} $TOTAL"
        done | xargs -n 3 -P "$PARALLEL" bash -c 'evaluate_dataset "$@"' _
    fi

    # Wait for all background jobs to complete
    wait

    print_info "All parallel jobs completed"
    echo ""
fi

# Aggregate individual logs into main log file
print_info "Aggregating logs..."
for dataset in "${DATASETS[@]}"; do
    log_file="$OUTPUT_DIR/${dataset}_log.txt"
    if [ -f "$log_file" ]; then
        cat "$log_file" >> "$LOG_FILE"
        rm "$log_file"  # Clean up individual log
    fi
done

# Count results
SUCCESSFUL=$(grep -c "^SUCCESS:" "$RESULTS_FILE" 2>/dev/null || echo 0)
FAILED=$(grep -c "^FAILED:" "$RESULTS_FILE" 2>/dev/null || echo 0)
SKIPPED=$(grep -c "^SKIPPED:" "$RESULTS_FILE" 2>/dev/null || echo 0)

# Print final summary
echo ""
echo "==================================================="
echo "Evaluation Complete at $(date)"
echo "==================================================="
print_info "Category:        $CATEGORY"
print_info "Total Datasets:  $TOTAL"
print_success "Successful:      $SUCCESSFUL"
print_error "Failed:          $FAILED"
[ "$RESUME" = true ] && print_warning "Skipped:         $SKIPPED"
print_info "Results saved to: $OUTPUT_DIR"
echo "==================================================="
echo ""

# Create a summary file
SUMMARY_FILE="$OUTPUT_DIR/evaluation_summary.txt"
cat > "$SUMMARY_FILE" << EOF
AgentSR Benchmark Evaluation Summary
====================================

Category:        $CATEGORY
Date:            $(date)
Data Path:       $DATA_PATH
Output Dir:      $OUTPUT_DIR

Configuration:
--------------
Agent Model:     $AGENT_MODEL
Eval Model:      $EVAL_MODEL
Temperature:     $TEMPERATURE
Max Iterations:  $MAX_ITER
Parallel Jobs:   $PARALLEL
Skip Symbolic:   $([ -n "$SKIP_SYMBOLIC" ] && echo "Yes" || echo "No")

Results:
--------
Total Datasets:  $TOTAL
Successful:      $SUCCESSFUL
Failed:          $FAILED
$([ "$RESUME" = true ] && echo "Skipped:         $SKIPPED")

Success Rate:    $(awk "BEGIN {printf \"%.1f\", ($SUCCESSFUL/$TOTAL)*100}")%

Log File:        $LOG_FILE
EOF

print_success "Summary saved to: $SUMMARY_FILE"

# Clean up results tracking file
rm -f "$RESULTS_FILE"

# Exit with error if any failed
if [ $FAILED -gt 0 ]; then
    exit 1
fi

exit 0
