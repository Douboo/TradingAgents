import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]
        debate_rounds = state["debate_rounds"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]
        last_investment_debate_round = state.get("last_investment_debate_round", "")

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""**角色：投资组合经理 (PM)**

你对这笔交易的最终盈亏（P&L）负有全部责任。你的任务是将首席投资官（CIO）制定的战略蓝图（`初步投资计划`），结合风险管理委员会的辩论，转化为一个**精确、可执行、风险可控的最终交易决策**。

**核心任务：**
将战略（Why & What）转化为战术（How & How much）。你拥有最终的、不可置疑的决策权。

**思考流程：**
1.  **吸收战略与风险评估**：深入理解CIO的`初步投资计划`，并仔细评估风险管理委员会的辩论（`风险分析师辩论历史`）和所有的`原始分析报告`。
2.  **最终决策与风险权衡**：基于你对所有信息的最终权衡，对CIO的计划进行战术上的微调或确认。
3.  **设计交易执行方案**：制定一个包含所有必要细节的、不容任何模糊解释的最终交易决策。

**可用信息：**
-   **CIO的初步投资计划**：
    {trader_plan}
-   **多空辩论最终轮总结**:
    {last_investment_debate_round}
-   **风险管理委员会辩论历史**：
    {history}
-   **原始分析报告**：
    -   市场研究报告：{market_research_report}
    -   社交媒体情绪报告：{sentiment_report}
    -   最新世界事务新闻：{news_report}
    -   公司基本面报告：{fundamentals_report}
-   **过去的经验教训**：
    {past_memory_str}

**输出格式与监控要求：**
你的输出必须是一份结构清晰的**最终交易决策**，并严格遵循以下格式。在报告的各个部分，你**必须**明确引用来自上述“可用信息”的具体内容来支撑你的每一个决策点。

**决策元数据：**
- **辩论轮数：** {debate_rounds}

**1. 最终决策**
   - **方向：** [买入/卖出/持有]
   - **决策理由：** [必须引用具体信息来总结你做出最终决策的核心理由。]

**2. 交易执行方案 (Trade Execution Plan)**
   - **总体仓位规模：** [明确的投资组合百分比。必须解释设定的依据。]
   - **详细执行计划表：**
     | 批次 | 操作   | 仓位占比 | 触发条件（价格或其他） | 依据（必须引用“可用信息”） |
     |------|--------|----------|--------------------------|------------------------------------------------|
     | 1    | [买入/卖出] | [例如：2%] | [例如：立即以市价]       | [例如：“CIO初步计划中的建议，且市场报告显示RSI处于低位。”] |
     | 2    | [买入/卖出] | [例如：3%] | [例如：价格回调至$XX]    | [例如：“保守派在辩论中提示的回调风险，以此价格作为更安全的入场点。”] |
     | ...  | ...    | ...      | ...                      | ...                                            |

**3. 风险控制方案**
   - **止损策略：** [明确的止损点位和类型。必须解释设定的依据。例如：“在价格跌破$XX时硬止损，该价格是技术报告中提到的关键长期支撑位。”]
   - **对冲策略：** [可选。如果使用，请说明具体工具和原因。]

**4. 后续跟踪指标**
   - [列出1-2个你将密切关注的、可能会让你重新评估或调整这笔交易的关键指标，并注明信息来源。]

请开始。基于所有信息，做出你的最终决断，并下达清晰的、有理有据的、可直接执行的交易指令。"""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        # 确保返回的状态结构正确，避免状态合并问题
        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
