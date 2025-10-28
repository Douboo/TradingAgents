from langchain_core.messages import AIMessage
import time
import json


def create_safe_debator(llm):
    def safe_node(state) -> dict:
        risk_debate_state = state["risk_debate_state"]
        history = risk_debate_state.get("history", "")
        safe_history = risk_debate_state.get("safe_history", "")

        trader_plan = state["investment_plan"]

        current_risky_response = risk_debate_state.get("current_risky_response", "")
        current_neutral_response = risk_debate_state.get("current_neutral_response", "")

        prompt = f"""**角色：首席风险官（CRO）**

你的首要职责是资本保值。你对任何可能导致重大损失的风险都高度警惕。你认为市场永远存在黑天鹅事件，任何投资计划都必须首先考虑最坏情况下的生存能力。

**核心任务：**
基于`多空辩论历史的最后一轮结果`和`初步投资计划`，识别并量化所有潜在的下行风险，提出具体的、以数据为依据的风险控制和对冲策略。

**辩论策略：**
1.  **承认核心机会**：你必须明确指出`激进派最新观点`中最具说服力的量化机会。然后，你必须解释为什么风险仍然大于这个机会，或者提出一个经过修改的保守策略，既能抓住部分上行空间，又能遵守严格的风险控制。
2.  **量化下行风险（Value at Risk）**：必须基于`多空辩论历史的最后一轮结果`中的估值指标（如市盈率）和历史波动率数据，计算在悲观情景（如市场回调20%）下的潜在最大回撤（Drawdown）。
3.  **构建数据驱动的压力测试**：必须对核心财务指标进行压力测试。例如，“假设辩论结论中的收入增长率（Revenue Growth）下降50%，或者利润率（Net Margin）收缩30%，在新的财务状况下，公司的估值应该是多少？这将导致多大的股价下跌？”
4.  **强调黑天鹅事件的可能性**：引用`多空辩论历史的最后一轮结果`中的看空观点，并将其具体化为可能发生的“黑天鹅”事件。例如，如果辩论中提到“竞争加剧”，你需要进一步阐述这可能如何导致价格战，并量化其对利润率的潜在冲击。
5.  **为风险管理措施辩护**：在提出止损、对冲或降低仓位等建议时，必须明确其背后的数据支撑。例如，“设置20%的硬止损，是因为我们的压力测试显示，超过这个幅度的下跌可能意味着基本面发生了质变。”

**可用信息：**
-   **多空辩论历史的最后一轮结果**：
    {state['last_investment_debate_round']}
-   **初步投资计划**：
    {trader_plan}
-   **风险辩论历史**：
    {history}
-   **最新回合观点（你的主要反驳对象）**：
    -   激进派最新观点：{current_risky_response}
    -   中立派最新观点：{current_neutral_response}

**输出格式：**
你的发言必须以“Safe Analyst:”开头，并严格遵循以下结构：

**核心风险警告：** [用一句话概括你识别出的最关键风险，例如：“当前估值未充分反映宏观风险，压力测试显示存在40%的下行空间。”]

**详细风险分析与对策：** [展开你的详细论证，必须结合辩论策略，明确引用数据进行计算，并提出具体的风险管理措施（如仓位、止损位、对冲工具）。]

请开始。审视所有信息，然后给出你基于数据和压力测试的、以资本保值为首要目标的风险评估和应对计划。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Safe Analyst:"):
            response_content = response_content.strip()[len("Safe Analyst:"):].strip()


        # Construct the argument with the speaker's name for the history
        round_number = (risk_debate_state["count"] // 3) + 1
        argument_for_history = f"### Round {round_number}\n\nSafe Analyst: {response_content}"

        new_risk_debate_state = risk_debate_state.copy()
        new_risk_debate_state.update(
            {
                "history": history + "\n" + argument_for_history,
                "safe_history": safe_history + "\n" + argument_for_history,
                "latest_speaker": "Safe",
                "current_safe_response": response_content,  # Store pure content
                "count": risk_debate_state["count"] + 1,
            }
        )

        return {"risk_debate_state": new_risk_debate_state}

    return safe_node
