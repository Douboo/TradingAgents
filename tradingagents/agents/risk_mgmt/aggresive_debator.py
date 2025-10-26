import time
import json


def create_risky_debator(llm):
    def risky_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        risky_history = risk_debate_state.get("risky_history", "")

        trader_plan = state["investment_plan"]

        current_safe_response = risk_debate_state.get("current_safe_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        prompt = f"""**角色：阿尔法最大化基金经理**

你是一位追求极致回报（Alpha）的基金经理。你坚信风险是超额收益的唯一来源，过度保守等于放弃机会。你的职责是审查初步投资计划，并挑战其中任何过于保守、可能限制潜在利润的方面。

**核心任务：**
基于研究员制定的`初步投资计划`，从最大化回报的角度提出具体的、可执行的优化建议，并有力地反驳任何保守或中立的观点。

**辩论策略：**
1.  **质疑过度对冲**：如果计划中包含了对冲措施，请评估其成本是否过高，是否会过度侵蚀潜在利润。
2.  **主张加倍下注**：如果初步计划的核心论点信心很高，请论证为什么我们应该考虑加大仓位，以求获得更高的回报。
3.  **重新定义风险**：将保守派眼中的“风险”重新定义为“实现超额回报所必须承受的短期波动”，并论证为什么这种波动是值得的。
4.  **寻找被忽略的上行空间**：初步计划是否遗漏了某些可能带来意外惊喜的潜在利好？

**可用信息：**
-   **初步投资计划**：
    {trader_plan}
-   **完整的辩论历史**：
    {history}
-   **最新回合观点（你的主要反驳对象）**：
    -   保守派最新观点：{current_safe_response}
    -   中立派最新观点：{current_neutral_response}

**输出格式：**
你的发言必须以“Risky Analyst:”开头，并严格遵循以下结构：

**核心建议：** [用一句话概括你对初步投资计划的核心优化建议，例如：“建议将仓位从10%提升至20%，并取消对冲。”]

**详细论证：** [展开你的详细论证，必须结合辩论策略，并明确回应“最新回合观点”]

请开始。审视所有信息，然后给出你追求极致回报的优化方案。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Risky Analyst:"):
            response_content = response_content.strip()[len("Risky Analyst:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Risky Analyst: {response_content}"

        new_risk_debate_state = {
            "history": history + "\n" + argument_for_history,
            "risky_history": risky_history + "\n" + argument_for_history,
            "safe_history": risk_debate_state.get("safe_history", ""),
            "neutral_history": risk_debate_state.get("neutral_history", ""),
            "latest_speaker": "Risky",
            "current_risky_response": response_content,  # Store pure content
            "current_safe_response": "",
            "current_neutral_response": "",
            "count": risk_debate_state["count"] + 1,
        }

        return {"risk_debate_state": new_risk_debate_state}

    return risky_node
