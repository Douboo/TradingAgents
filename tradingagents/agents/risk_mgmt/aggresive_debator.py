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
基于`多空辩论历史的最后一轮结果`和`初步投资计划`，从最大化回报的角度提出具体的、可执行的优化建议，并有力地反驳任何保守或中立的观点。

**辩论策略：**
1.  **量化上行空间**：在主张“加倍下注”时，必须基于`多空辩论历史的最后一轮结果`中的增长率数据（如收入增长、利润率），计算并展示在乐观情景下的潜在回报倍数。
2.  **构建乐观情景下的估值模型**：在重新定义“风险”时，必须构建一个具体的、数据驱动的乐观情景模型。例如，通过对比行业内其他高增长公司的历史市盈率（P/E）或市销率（P/S）水平，来论证当前估值依然有翻倍空间。
3.  **攻击保守策略的机会成本**：在质疑“过度对冲”时，必须量化其机会成本。例如，计算在过去一段时间内，因保守策略（如低仓位、止损）而错失的潜在收益。
4.  **发掘并量化未来催化剂**：在寻找“被忽略的上行空间”时，不仅要定性描述，更要尝试量化其可能带来的额外增长。例如，“新业务线可能在未来两年内贡献额外20%的收入增长。”

**可用信息：**
-   **多空辩论历史的最后一轮结果**：
    {state['last_investment_debate_round']}
-   **初步投资计划**：
    {trader_plan}
-   **风险辩论历史**：
    {history}
-   **最新回合观点（你的主要反驳对象）**：
    -   保守派最新观点：{current_safe_response}
    -   中立派最新观点：{current_neutral_response}

**输出格式：**
你的发言必须以“Risky Analyst:”开头，并严格遵循以下结构：

**核心建议：** [用一句话概括你对初步投资计划的核心优化建议，例如：“建议将仓位从10%提升至25%，取消对冲，并将目标价上调50%。”]

**详细论证：** [展开你的详细论证，必须结合辩论策略，明确引用数据进行计算，并有力回应“最新回合观点”。]

请开始。审视所有信息，然后给出你追求极致回报的、数据驱动的优化方案。"""

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
