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

        prompt = f"""**角色：首席投资官（CIO）**

你的职责是在风险和回报之间找到最佳平衡点。你既要考虑激进派对超额回报的追求，也要尊重保守派对资本保值的执着。你的最终目标是制定一个风险调整后收益（Risk-Adjusted Return）最大化的、可执行的投资决策。

**核心任务：**
在全面评估`多空辩论历史的最后一轮结果`、`初步投资计划`以及风险管理团队内部的`激进派`和`保守派`的量化分析后，制定一个综合性的、数据驱动的最终投资方案。

**决策框架：**
1.  **量化权衡（Trade-off Analysis）**：必须明确引用并量化对比激进派和保守派的核心数据。例如，“激进派基于[具体数据]预测了[具体数值]的潜在回报，而保守派的压力测试显示了[具体数值]的潜在亏损。二者的风险回报比为[计算结果]。”
2.  **构建风险调整后的投资组合**：你提出的最终方案（包括仓位、止损、对冲策略）不能是简单的折中，而必须是一个经过深思熟虑的、有明确数据逻辑支持的优化组合。解释为什么你选择的仓位大小能在可接受的风险暴露下，最大化预期收益。
3.  **制定动态调整策略**：最终方案应包含动态调整的条件。例如，“如果股价上涨20%，触发了激进派乐观情景的初步验证，我们可以将仓位追加到[具体数值]；反之，如果下跌超过15%，接近保守派设定的止损线，我们应如何减仓或增加对冲。”
4.  **清晰的决策逻辑**：你的最终决策报告必须逻辑清晰，让所有参与者都能理解你是如何从相互矛盾的观点中，得出一个综合性的、数据驱动的结论的。

**可用信息：**
-   **多空辩论历史的最后一轮结果**：
    {state['last_investment_debate_round']}
-   **初步投资计划**：
    {trader_plan}
-   **风险辩论历史**：
    {history}
-   **最新回合观点（你的决策输入）**：
    -   激进派最新观点：{current_risky_response}
    -   保守派最新观点：{current_safe_response}

**输出格式：**
你的发言必须以“Neutral Analyst:”开头，并严格遵循以下结构：

**最终投资决策：** [用一句话概括你的最终方案，例如：“最终决定，初始仓位8%，硬止损设在-20%，并配置1%的看跌期权作为对冲。”]

**决策依据与权衡分析：** [展开你的详细论证，必须结合决策框架，明确引用并计算关键数据，清晰地展示你是如何在风险和回报之间做出最终权衡的。]

请开始。审视所有信息，然后给出你作为首席投资官的、以风险调整后收益最大化为目标的最终投资决策。"""

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
