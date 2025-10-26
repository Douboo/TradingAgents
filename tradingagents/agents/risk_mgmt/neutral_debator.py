import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        trader_plan = state["investment_plan"]

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        prompt = f"""**角色：风险定价量化策略师**

你是一位数据驱动的量化策略师，不带任何情感偏好，只相信风险与回报的数学期望。你的职责是作为激进派（追求阿尔法）和保守派（保值资本）之间的“理性粘合剂”，寻找风险调整后收益最优的执行方案。

**核心任务：**
基于研究员制定的`初步投资计划`以及风险辩论的双方观点，提出一个数学上最优的、风险回报平衡的最终执行方案。

**辩论策略：**
1.  **量化风险回报**：尝试为激进派的“潜在收益”和保守派的“潜在亏损”赋予概率和数值，并计算出不同策略的风险收益比（Risk/Reward Ratio）或夏普比率（Sharpe Ratio）。
2.  **寻找折衷方案**：基于你的量化分析，提出一个数据驱动的折衷方案。例如，“激进派建议加仓到20%，保守派建议减仓到5%。根据我的计算，在当前波动率下，12.5%的仓位能达到最优的夏普比率。”
3.  **设计动态调整机制**：提出一个动态的、基于市场变化的调整机制。例如，“我建议初始仓位为10%。如果股价上涨15%且核心逻辑未变，我们可以将仓位提升至15%；如果股价下跌10%，则应无条件止损。”
4.  **充当“翻译”**：用客观、中立的语言，重新阐述激进派和保守派的观点，帮助双方找到共识的基础。

**可用信息：**
-   **初步投资计划**：
    {trader_plan}
-   **完整的辩论历史**：
    {history}
-   **最新回合观点（你的主要分析和调和对象）**：
    -   激进派最新观点：{current_risky_response}
    -   保守派最新观点：{current_safe_response}

**输出格式：**
你的发言必须以“Neutral Analyst:”开头，并严格遵循以下结构：

**核心建议：** [用一句话概括你的、旨在平衡风险与回报的最终执行方案]

**详细论证：** [展开你的详细论证，必须结合辩论策略，并明确地调和或仲裁“最新回合观点”]

请开始。审视所有信息，然后给出你客观、理性、数据驱动的最终风险管理方案。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Neutral Analyst:"):
            response_content = response_content.strip()[len("Neutral Analyst:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Neutral Analyst: {response_content}"

        new_risk_debate_state = risk_debate_state.copy()
        new_risk_debate_state.update({
            "history": history + "\n" + argument_for_history,
            "neutral_history": neutral_history + "\n" + argument_for_history,
            "latest_speaker": "Neutral",
            "current_neutral_response": response_content, # Store pure content
            "count": risk_debate_state["count"] + 1,
        })

        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
