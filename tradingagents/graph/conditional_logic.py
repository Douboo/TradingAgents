# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue based on arbiter's decision or max rounds."""
        investment_debate_state = state.get("investment_debate_state", {})
        count = investment_debate_state.get("count", 0)
        
        print(f"--- DEBUG: should_continue_debate - current count: {count}, max_rounds: {self.max_debate_rounds}")
        
        # End debate if max rounds are reached
        if count >= self.max_debate_rounds * 2: # 2 speakers per round
            print(f"--- DEBUG: Debate decision -> Research Manager (max rounds reached)")
            return "Research Manager"

        # End debate if arbiter decides to
        if state.get("debate_arbiter_decision") == "end":
            print(f"--- DEBUG: Debate decision -> Research Manager (arbiter ended)")
            return "Research Manager"
        
        # Alternate between Bull and Bear researchers
        latest_speaker = investment_debate_state.get("latest_speaker")
        if latest_speaker == "Bull":
            decision = "Bear Researcher"
        else:  # Bear or initial state
            decision = "Bull Researcher"
        print(f"--- DEBUG: Debate decision -> {decision}")
        return decision

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue based on arbiter's decision or max rounds."""
        risk_debate_state = state.get("risk_debate_state", {})
        count = risk_debate_state.get("count", 0)
        
        print(f"--- DEBUG: should_continue_risk_analysis - current count: {count}, max_rounds: {self.max_risk_discuss_rounds}")

        if count >= self.max_risk_discuss_rounds * 3: # 3 speakers per round
            print(f"--- DEBUG: Risk analysis decision -> Risk Judge (max rounds reached)")
            return "Risk Judge"

        if state.get("risk_arbiter_decision") == "end":
            print(f"--- DEBUG: Risk analysis decision -> Risk Judge (arbiter ended)")
            return "Risk Judge"
        else:
            # This logic needs to be more robust to handle the debate flow
            # For now, we just alternate, but a better approach would be a round-robin
            # or based on who needs to respond.
            latest_speaker = risk_debate_state.get("latest_speaker")
            if latest_speaker == "Risky":
                decision = "Safe Analyst"
            elif latest_speaker == "Safe":
                decision = "Neutral Analyst"
            else: # Neutral or initial state
                decision = "Risky Analyst"
            print(f"--- DEBUG: Risk analysis decision -> {decision}")
            return decision
