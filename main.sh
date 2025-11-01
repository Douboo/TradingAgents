#!/bin/bash

# TradingAgents Automated Analysis Script
# Automated version for background execution with support for multiple tickers.

# --- Conda Environment Setup ---
# Ensure conda is initialized
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    echo "警告: 无法找到conda初始化脚本。"
fi

# Check if conda command is available
if ! command -v conda &> /dev/null; then
    echo "错误: Conda未安装或未正确初始化。"
    echo "请先运行 'conda init' 并重启终端。"
    exit 1
fi

# Activate conda environment
conda activate tradingagents
if [ $? -ne 0 ]; then
    echo "错误: 无法激活 'tradingagents' conda环境。"
    echo "请确保环境存在，或使用 'conda create -n tradingagents python=3.11' 创建。"
    exit 1
fi

# --- Default Parameters ---
TICKERS="baba,tcehy" # Default ticker
DATE=$(date +%Y-%m-%d)
ANALYSTS="market social news fundamentals"
DEPTH=3
PROVIDER="deepseek"
SHALLOW_MODEL="deepseek-chat"
DEEP_MODEL="deepseek-reasoner"
QUIET=false
NO_SAVE=false

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--tickers)
            TICKERS="$2"
            shift 2
            ;;
        -d|--date)
            DATE="$2"
            shift 2
            ;;
        -a|--analysts)
            ANALYSTS="$2"
            shift 2
            ;;
        -r|--depth)
            DEPTH="$2"
            shift 2
            ;;
        -p|--provider)
            PROVIDER="$2"
            shift 2
            ;;
        --shallow-model)
            SHALLOW_MODEL="$2"
            shift 2
            ;;
        --deep-model)
            DEEP_MODEL="$2"
            shift 2
            ;;
        -q|--quiet)
            QUIET=true
            shift
            ;;
        --no-save)
            NO_SAVE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -t, --tickers SYMBOLS   Comma-separated ticker symbols to analyze (e.g., \"QQQ,SPY,AAPL\")."
            echo "  -d, --date DATE         Analysis date in YYYY-MM-DD format (default: today)."
            echo "  -a, --analysts LIST     Space-separated list of analysts (default: \"market social news fundamentals\")."
            echo "  -r, --depth LEVEL       Research depth: 1=Shallow, 3=Medium, 5=Deep (default: 1)."
            echo "  -p, --provider NAME     LLM provider (default: deepseek)."
            echo "  --shallow-model MODEL   Model for shallow analysis (default: deepseek-chat)."
            echo "  --deep-model MODEL      Model for deep analysis (default: deepseek-reasoner)."
            echo "  -q, --quiet             Quiet mode (minimal output)."
            echo "  --no-save               Do not save reports to files."
            echo "  -h, --help              Show this help message."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# --- Main Execution Loop ---
echo "启动TradingAgents自动化分析系统..."
echo "待分析股票列表: $TICKERS"
echo "----------------------------------------"

# Convert comma-separated string to an array to handle tickers
IFS=',' read -r -a TICKER_ARRAY <<< "$TICKERS"

# Loop through each ticker and run the analysis
for TICKER in "${TICKER_ARRAY[@]}"; do
    # Trim whitespace from ticker
    TICKER=$(echo "$TICKER" | xargs)
    if [ -z "$TICKER" ]; then
        continue
    fi

    echo ""
    echo "▶️  开始分析: $TICKER"
    echo "========================================="

    # Build command arguments for the current ticker using an array
    ARGS=(-t "$TICKER" -d "$DATE" -r "$DEPTH" -p "$PROVIDER" --shallow-model "$SHALLOW_MODEL" --deep-model "$DEEP_MODEL")
    
    # Handle multi-word analyst list correctly
    IFS=' ' read -r -a ANALYST_ARRAY <<< "$ANALYSTS"
    ARGS+=(-a "${ANALYST_ARRAY[@]}")

    if [ "$QUIET" = true ]; then
        ARGS+=("-q")
    fi

    if [ "$NO_SAVE" = true ]; then
        ARGS+=("--no-save")
    fi

    echo "执行参数: python main.py ${ARGS[*]}"
    
    # Run the main python script
    python main.py "${ARGS[@]}"
    
    if [ $? -eq 0 ]; then
        echo "✅  成功完成分析: $TICKER"
    else
        echo "❌  分析失败: $TICKER"
    fi
    echo "========================================="
done

echo ""
echo "所有股票分析任务已完成。"