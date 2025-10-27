import time
import json


def create_neutral_debator(llm):
    def neutral_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        count = risk_debate_state.get("count", 0)
        print(f"--- DEBUG (Round {count}): Entering Neutral Analyst")
        
        history = risk_debate_state.get("history", "")
        neutral_history = risk_debate_state.get("neutral_history", "")

        trader_plan = state["investment_plan"]

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_safe_response = risk_debate_state.get("current_safe_response", "")

        prompt = f"""**角色：首席投资官 (CIO)**

你的座右铭是"在风险和回报之间找到最佳平衡点。"你的职责是综合激进派和保守派的观点，制定一个平衡的、可执行的投资决策。

**核心任务：**
基于研究员制定的`初步投资计划`，以及激进派和保守派的辩论，制定一个平衡的、可执行的投资决策。

**辩论策略：**
1.  **综合观点**：识别激进派和保守派观点中的合理部分，并尝试将它们融合成一个连贯的策略。
2.  **寻求妥协**：在激进派和保守派的分歧点上，提出折中方案。例如，如果激进派建议全仓买入，而保守派建议观望，你可以建议先建仓50%。
3.  **强调执行细节**：关注策略的实际执行层面，包括具体的仓位管理、入场点、止损点和目标价位。
4.  **量化决策依据**：用数据和逻辑支持你的决策，避免模糊的定性描述。

**可用信息：**
-   **初步投资计划**：
    {trader_plan}
-   **完整的辩论历史**：
    {history}
-   **最新回合观点（你的主要综合对象）**：
    -   激进派最新观点：{current_risky_response}
    -   保守派最新观点：{current_safe_response}

**输出格式：**
你的发言必须以"Neutral Analyst:"开头，并严格遵循以下结构：

**最终决策：** [用一句话概括你的最终投资决策，例如："建议以当前价格建仓60%，设置-8%止损和+15%止盈。"]

**详细论证：** [展开你的详细论证，必须结合辩论策略，并明确回应"最新回合观点"]

请开始。审视所有信息，然后给出你平衡风险与回报的最终投资决策。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Neutral Analyst:"):
            response_content = response_content.strip()[len("Neutral Analyst:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Neutral Analyst: {response_content}"

        new_risk_debate_state = risk_debate_state.copy()
        new_risk_debate_state.update(
            {
                "history": history + "\n" + argument_for_history,
                "neutral_history": neutral_history + "\n" + argument_for_history,
                "latest_speaker": "Neutral",
                "current_neutral_response": response_content,  # Store pure content
                "count": risk_debate_state["count"] + 1,
            }
        )

        print(f"--- DEBUG (Round {count}): Neutral Analyst new history length: {len(new_risk_debate_state['history'])}")
        return {"risk_debate_state": new_risk_debate_state}

    return neutral_node
