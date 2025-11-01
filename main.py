#!/usr/bin/env python3
"""
TradingAgents Automated Analysis Script
Automated version of the CLI for background execution
"""

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
import datetime
import argparse
import sys
from pathlib import Path
from cli.models import AnalystType
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def create_automated_selections(ticker=None, date=None, analysts=None, 
                               research_depth=None, llm_provider=None,
                               shallow_thinker=None, deep_thinker=None):
    """
    Create automated selections based on provided parameters or defaults
    """
    # Default values
    selections = {
        "ticker": ticker or "QQQ",
        "analysis_date": date or datetime.datetime.now().strftime("%Y-%m-%d"),
        "analysts": analysts or [AnalystType.MARKET, AnalystType.SOCIAL, 
                                 AnalystType.NEWS, AnalystType.FUNDAMENTALS],
        "research_depth": research_depth or 1,
        "llm_provider": llm_provider or "deepseek",
        "backend_url": "https://api.deepseek.com/v1",
        "shallow_thinker": shallow_thinker or "deepseek-chat",
        "deep_thinker": deep_thinker or "deepseek-reasoner"
    }
    
    return selections

def run_automated_analysis(selections, debug=True, save_reports=True):
    """
    Run automated analysis with the provided selections
    """
    # Create config with selected research depth
    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = selections["research_depth"]
    config["max_risk_discuss_rounds"] = selections["research_depth"]
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()

    # Initialize the graph
    ta = TradingAgentsGraph(
        [analyst.value for analyst in selections["analysts"]], 
        config=config, 
        debug=debug
    )

    # Create result directory if saving reports
    if save_reports:
        results_dir = Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
        results_dir.mkdir(parents=True, exist_ok=True)
        report_dir = results_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        print(f"Reports will be saved to: {report_dir}")

    # Run analysis
    print(f"Starting analysis for {selections['ticker']} on {selections['analysis_date']}")
    print(f"Analysts: {', '.join([analyst.value for analyst in selections['analysts']])}")
    print(f"Research Depth: {selections['research_depth']}")
    print(f"LLM Provider: {selections['llm_provider']}")
    
    try:
        # Forward propagate
        final_state, decision = ta.propagate(selections["ticker"], selections["analysis_date"])
        
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Final Decision: {decision}")
        
        # Save reports if requested
        if save_reports:
            # Save each report section
            for section_name in ["market_report", "sentiment_report", "news_report", 
                               "fundamentals_report", "investment_plan", 
                               "trader_investment_plan", "final_trade_decision"]:
                if section_name in final_state and final_state[section_name]:
                    file_path = report_dir / f"{section_name}.md"
                    with open(file_path, "w") as f:
                        f.write(final_state[section_name])
                    print(f"Saved: {file_path}")
        
        return decision
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

def main():
    """
    Main function with command line argument support
    """
    parser = argparse.ArgumentParser(description="TradingAgents Automated Analysis")
    parser.add_argument("--ticker", "-t", default="QQQ", 
                       help="Ticker symbol to analyze (default: QQQ)")
    parser.add_argument("--date", "-d", 
                       default=datetime.datetime.now().strftime("%Y-%m-%d"),
                       help="Analysis date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--analysts", "-a", nargs="+", 
                       choices=["market", "social", "news", "fundamentals"],
                       default=["market", "social", "news", "fundamentals"],
                       help="Analysts to include (default: all)")
    parser.add_argument("--depth", "-r", type=int, default=3, 
                       choices=[1, 3, 5],
                       help="Research depth: 1=Shallow, 3=Medium, 5=Deep (default: 1)")
    parser.add_argument("--provider", "-p", default="deepseek",
                       choices=["deepseek", "openai", "anthropic", "google", "openrouter", "ollama"],
                       help="LLM provider (default: deepseek)")
    parser.add_argument("--shallow-model", default="deepseek-chat",
                       help="Quick-thinking model (default: deepseek-chat)")
    parser.add_argument("--deep-model", default="deepseek-reasoner",
                       help="Deep-thinking model (default: deepseek-reasoner)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save reports to files")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode (minimal output)")
    
    args = parser.parse_args()
    
    # Convert analyst strings to AnalystType enum
    analyst_mapping = {
        "market": AnalystType.MARKET,
        "social": AnalystType.SOCIAL,
        "news": AnalystType.NEWS,
        "fundamentals": AnalystType.FUNDAMENTALS
    }
    analysts = [analyst_mapping[a] for a in args.analysts]
    
    # Create automated selections
    selections = create_automated_selections(
        ticker=args.ticker,
        date=args.date,
        analysts=analysts,
        research_depth=args.depth,
        llm_provider=args.provider,
        shallow_thinker=args.shallow_model,
        deep_thinker=args.deep_model
    )
    
    # Run analysis
    decision = run_automated_analysis(
        selections, 
        debug=not args.quiet,
        save_reports=not args.no_save
    )
    
    if decision:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Error

if __name__ == "__main__":
    main()
