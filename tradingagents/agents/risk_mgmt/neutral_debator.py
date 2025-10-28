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

**决策框架（重要！必须严格遵守）：**
1.  **历史锚定原则 (Historical Anchor Principle)**: 你的首要任务是**锚定第一轮的辩论共识**。后续每一轮的决策都应被视为对第一轮决策的微调和优化，而不是推倒重来。任何偏离第一轮核心逻辑的重大调整，都必须有新出现的、强有力的基本面数据或事件作为支撑。你必须明确指出当前决策与第一轮决策的差异，并解释其合理性。
2.  **识别论点升级与收敛 (Identify Escalation and Convergence)**: 你必须审视激进派和保守派的论点是否比前几轮出现了不合理的“升级”或“极端化”。更重要的是，你必须**识别并放大双方的“共识收敛点”**：即激进派开始承认并对冲的风险，以及保守派开始认可并接受的机会。这些收敛点是你构建最终决策的基石，而不是在两个极端观点之间简单取中。
3.  **独立量化与权衡 (Independent Quantitative Analysis)**:
    -   **禁止简单平均**: 严禁对激进派和保守派提出的核心量化指标（如夏普比率、预期收益、潜在亏损）进行简单的数学平均。你的独立判断和计算必须**优先考虑并基于“共识收敛点”**，选择你认为更合理的模型作为基准。
    -   **量化对比**: 必须明确引用并量化对比激进派和保守派的核心数据。例如，“激进派基于[具体数据]预测了[具体数值]的潜在回报，而保守派的压力测试显示了[具体数值]的潜在亏损。经过我的独立评估，我认为一个更合理的风险回报比是[计算结果]。”
4.  **构建风险调整后的投资组合**: 基于你的独立判断和历史锚定原则，构建一个经过深思熟虑的优化组合（包括仓位、止损、对冲策略）。解释为什么你选择的仓位大小能在可接受的风险暴露下，最大化预期收益，并说明这与第一轮的决策逻辑如何保持一致或为何需要调整。
5.  **制定动态调整策略**: 最终方案应包含动态调整的条件。例如，“如果股价上涨20%，触发了激进派乐观情景的初步验证，我们可以将仓位追加到[具体数值]；反之，如果下跌超过15%，接近保守派设定的止损线，我们应如何减仓或增加对冲。”
6.  **清晰的决策逻辑**: 你的最终决策报告必须逻辑清晰，让所有参与者都能理解你是如何从相互矛盾的观点中，得出一个独立的、有数据支持的、且与历史决策连贯的结论。

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
        round_number = (risk_debate_state["count"] // 3) + 1
        argument_for_history = f"### Round {round_number}\n\nNeutral Analyst: {response_content}"

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
