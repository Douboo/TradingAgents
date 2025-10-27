# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            self.deep_thinking_llm = ChatOpenAI(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatOpenAI(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        elif self.config["llm_provider"].lower() == 'deepseek':
            self.deep_thinking_llm = ChatDeepSeek(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatDeepSeek(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        print(f"--- DEBUG: Config max_debate_rounds: {self.config.get('max_debate_rounds', 'NOT_FOUND')}")
        print(f"--- DEBUG: Config max_risk_discuss_rounds: {self.config.get('max_risk_discuss_rounds', 'NOT_FOUND')}")
        print(f"--- DEBUG: Config keys containing 'max': {[k for k in self.config.keys() if 'max' in k]}")
        print(f"--- DEBUG: Config keys containing 'debate': {[k for k in self.config.keys() if 'debate' in k]}")
        print(f"--- DEBUG: Config keys containing 'round': {[k for k in self.config.keys() if 'round' in k]}")
        
        # 确保配置参数正确传递
        debate_rounds = self.config.get("max_debate_rounds", 2)
        risk_rounds = self.config.get("max_risk_discuss_rounds", 2)
        
        print(f"--- DEBUG: Passing to ConditionalLogic - debate_rounds: {debate_rounds}, risk_rounds: {risk_rounds}")
        
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=debate_rounds,
            max_risk_discuss_rounds=risk_rounds,
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        print(f"--- DEBUG: State before entering graph: {init_agent_state.keys()}")
        print(f"--- DEBUG: Investment debate state before graph: {init_agent_state.get('investment_debate_state', {}).keys()}")
        print(f"--- DEBUG: Risk debate state before graph: {init_agent_state.get('risk_debate_state', {}).keys()}")

        # Execute the graph
        if self.debug:
            # Debug mode with tracing
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk.get("messages", [])) == 0:
                    pass
                else:
                    # chunk["messages"][-1].pretty_print()
                    pass
                trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        print(f"--- DEBUG: State after exiting graph: {final_state.keys()}")
        print(f"--- DEBUG: Investment debate state after graph: {final_state.get('investment_debate_state', {}).keys()}")
        print(f"--- DEBUG: Risk debate state after graph: {final_state.get('risk_debate_state', {}).keys()}")

        # DEBUG: 检查辩论历史是否存在
        investment_history = final_state.get('investment_debate_state', {}).get('history', '')
        risk_history = final_state.get('risk_debate_state', {}).get('history', '')
        print(f"--- DEBUG: Investment debate history exists: {bool(investment_history)}")
        print(f"--- DEBUG: Risk debate history exists: {bool(risk_history)}")
        print(f"--- DEBUG: Investment debate history length: {len(investment_history)}")
        print(f"--- DEBUG: Risk debate history length: {len(risk_history)}")
        
        # DEBUG: 检查history内容是否为空或异常
        print(f"--- DEBUG: Investment debate history is empty: {not investment_history.strip()}")
        print(f"--- DEBUG: Risk debate history is empty: {not risk_history.strip()}")
        
        # DEBUG: 检查history内容的前200个字符
        if investment_history:
            print(f"--- DEBUG: Investment debate history first 200 chars: {investment_history[:200]}")
        else:
            print("--- DEBUG: Investment debate history is EMPTY")
            
        if risk_history:
            print(f"--- DEBUG: Risk debate history first 200 chars: {risk_history[:200]}")
        else:
            print("--- DEBUG: Risk debate history is EMPTY")

        # Store current state for reflection
        self.curr_state = final_state

        # DEBUG: 准备进入文件写入阶段
        print("--- DEBUG: Preparing to enter _log_state method for file writing")

        # Log state
        self._log_state(trade_date, final_state)

        # DEBUG: 文件写入完成
        print("--- DEBUG: File writing completed successfully")

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        
        # DEBUG: 进入文件写入方法
        print("--- DEBUG: Entering _log_state method")
        print(f"--- DEBUG: Ticker: {self.ticker}, Trade date: {trade_date}")
        
        # DEBUG: 检查最终状态的关键字段
        print(f"--- DEBUG: Final state keys: {final_state.keys()}")
        print(f"--- DEBUG: Investment debate state in final_state: {'investment_debate_state' in final_state}")
        print(f"--- DEBUG: Risk debate state in final_state: {'risk_debate_state' in final_state}")
        
        if 'investment_debate_state' in final_state:
            investment_state = final_state['investment_debate_state']
            print(f"--- DEBUG: Investment debate state keys: {investment_state.keys()}")
            print(f"--- DEBUG: Investment debate state has history: {'history' in investment_state}")
            if 'history' in investment_state:
                print(f"--- DEBUG: Investment debate history content length: {len(investment_state['history'])}")
                print(f"--- DEBUG: Investment debate history first 200 chars: {investment_state['history'][:200]}")
        
        if 'risk_debate_state' in final_state:
            risk_state = final_state['risk_debate_state']
            print(f"--- DEBUG: Risk debate state keys: {risk_state.keys()}")
            print(f"--- DEBUG: Risk debate state has history: {'history' in risk_state}")
            if 'history' in risk_state:
                print(f"--- DEBUG: Risk debate history content length: {len(risk_state['history'])}")
        
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # DEBUG: 准备创建报告目录
        print("--- DEBUG: Preparing to create reports directory")
        
        # Save individual reports and debate histories to markdown files
        reports_dir = Path(f"results/{self.ticker}/{trade_date}/reports")
        
        # DEBUG: 检查目录创建
        print(f"--- DEBUG: Reports directory path: {reports_dir}")
        print(f"--- DEBUG: Reports directory exists before creation: {reports_dir.exists()}")
        
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"--- DEBUG: Reports directory exists after creation: {reports_dir.exists()}")
        print(f"--- DEBUG: Reports directory is directory: {reports_dir.is_dir()}")
        
        report_files = {
            "market_report.md": final_state.get("market_report", ""),
            "sentiment_report.md": final_state.get("sentiment_report", ""),
            "news_report.md": final_state.get("news_report", ""),
            "fundamentals_report.md": final_state.get("fundamentals_report", ""),
            "investment_plan.md": final_state.get("investment_plan", ""),
            "final_trade_decision.md": final_state.get("final_trade_decision", ""),
            "investment_debate_history.md": final_state.get("investment_debate_state", {}).get("history", ""),
            "risk_debate_history.md": final_state.get("risk_debate_state", {}).get("history", ""),
        }
        
        # DEBUG: 检查每个文件的内容
        print("--- DEBUG: Checking content of each report file")
        for filename, content in report_files.items():
            content_exists = bool(content and content.strip())
            print(f"--- DEBUG: {filename} - content exists: {content_exists}, length: {len(content)}")
            if filename == "investment_debate_history.md":
                print(f"--- DEBUG: {filename} first 200 chars: {content[:200] if content else 'EMPTY'}")

        # DEBUG: 开始文件写入
        print("--- DEBUG: Starting file writing process")
        
        for filename, content in report_files.items():
            file_path = reports_dir / filename
            print(f"--- DEBUG: Writing {filename} to {file_path}")
            print(f"--- DEBUG: {filename} content length: {len(content)}")
            
            try:
                # DEBUG: 检查文件路径
                print(f"--- DEBUG: File path absolute: {file_path.absolute()}")
                print(f"--- DEBUG: Parent directory exists: {file_path.parent.exists()}")
                print(f"--- DEBUG: Parent directory is directory: {file_path.parent.is_dir()}")
                
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"--- DEBUG: Successfully wrote {filename}")
                
                # DEBUG: 验证文件是否真的写入
                written_file_exists = file_path.exists()
                written_file_size = file_path.stat().st_size if written_file_exists else 0
                print(f"--- DEBUG: {filename} file exists after write: {written_file_exists}")
                print(f"--- DEBUG: {filename} file size: {written_file_size} bytes")
                
            except Exception as e:
                print(f"--- DEBUG: Error writing {filename}: {e}")
                import traceback
                print(f"--- DEBUG: Error traceback: {traceback.format_exc()}")

        # DEBUG: 文件写入完成
        print("--- DEBUG: File writing process completed")

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
