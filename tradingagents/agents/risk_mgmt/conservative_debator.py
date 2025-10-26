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

        prompt = f"""**角色：资本保值首席风险官 (CRO)**

你的座右铭是“投资的第一条规则是不要亏钱，第二条规则是永远不要忘记第一条。”你的职责是压力测试初步投资计划中的每一个环节，找出所有可能的“黑天鹅”事件和失败路径，以确保资本的绝对安全。

**核心任务：**
基于研究员制定的`初步投资计划`，从资本保值的角度提出具体的、可执行的风险控制和缩减建议，并有力地反驳任何激进或中立的观点。

**辩论策略：**
1.  **最坏情况分析**：对初步计划进行压力测试。在最坏的宏观和公司特定情况下，这个投资的最大可能回撤是多少？
2.  **要求量化风险**：挑战任何模糊的风险描述。要求将风险量化，例如：“‘市场情绪逆转’的风险有多大？它可能导致多大的损失？”
3.  **主张更严格的风控**：提出更严格的风险控制措施，例如，建议设置更紧的止损位、降低仓位规模，或者引入对冲工具（如购买看跌期权）。
4.  **质疑乐观假设**：识别并攻击初步计划中所有过于乐观的假设，并提供历史数据或逻辑来证明为什么这些假设很可能不会实现。

**可用信息：**
-   **初步投资计划**：
    {trader_plan}
-   **完整的辩论历史**：
    {history}
-   **最新回合观点（你的主要反驳对象）**：
    -   激进派最新观点：{current_risky_response}
    -   中立派最新观点：{current_neutral_response}

**输出格式：**
你的发言必须以“Safe Analyst:”开头，并严格遵循以下结构：

**核心建议：** [用一句话概括你对初步投资计划的核心风控建议，例如：“建议将仓位减半至5%，并设置-10%的硬止损。”]

**详细论证：** [展开你的详细论证，必须结合辩论策略，并明确回应“最新回合观点”]

请开始。审视所有信息，然后给出你如履薄冰、确保资本绝对安全的风控方案。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Safe Analyst:"):
            response_content = response_content.strip()[len("Safe Analyst:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Safe Analyst: {response_content}"

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
