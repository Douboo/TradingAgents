# TradingAgents/graph/propagation.py

from typing import Dict, Any
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)


class Propagator:
    """Handles state initialization and propagation through the graph."""

    def __init__(self, max_recur_limit=100):
        """Initialize with configuration parameters."""
        self.max_recur_limit = max_recur_limit

    def create_initial_state(
        self, company_name: str, trade_date: str
    ) -> Dict[str, Any]:
        """Create the initial state for the agent graph."""
        print(f"--- DEBUG: Creating initial state for risk_debate_state with count: 0")
        return {
            "messages": [("human", company_name)],
            "company_of_interest": company_name,
            "trade_date": str(trade_date),
            "investment_debate_state": InvestDebateState(
                {"history": "", "current_response": "", "count": 0}
            ),
            "risk_debate_state": RiskDebateState(
                {
                    "history": "",
                    "current_risky_response": "",
                    "current_safe_response": "",
                    "current_neutral_response": "",
                    "count": 0,
                }
            ),
            "market_report": "",
            "fundamentals_report": "",
            "sentiment_report": "",
            "news_report": "",
        }

    def get_graph_args(self) -> Dict[str, Any]:
        """Get arguments for the graph invocation."""
        return {
            "stream_mode": "values",
            "config": {"recursion_limit": self.max_recur_limit},
        }

    def extract_last_round(self, state: AgentState) -> dict:
        """Extracts the last round of the investment debate from the history."""
        history = state.get("investment_debate_state", {}).get("history", "")
        if not history.strip():
            return {"last_investment_debate_round": "No debate history yet."}

        rounds = history.strip().split('\n\n')
        last_round_discussions = []
        
        # A round consists of a bull and a bear statement
        bull_prefix = "Bull Analyst:"
        bear_prefix = "Bear Analyst:"
        
        # Find the last bull and bear statements
        last_bull = None
        last_bear = None
        
        for i in range(len(rounds) - 1, -1, -1):
            if rounds[i].startswith(bear_prefix) and not last_bear:
                last_bear = rounds[i]
            elif rounds[i].startswith(bull_prefix) and not last_bull:
                last_bull = rounds[i]
            
            if last_bull and last_bear:
                break
        
        if last_bull:
            last_round_discussions.append(last_bull)
        if last_bear:
            last_round_discussions.append(last_bear)

        if not last_round_discussions:
            return {"last_investment_debate_round": "Could not extract the last round."}

        return {"last_investment_debate_round": "\n\n".join(last_round_discussions)}
