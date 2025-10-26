import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        debate_rounds = state["debate_rounds"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        debate_summary = state.get("debate_summary", "") # 从仲裁者获取的摘要

        prompt = f"""**角色：首席投资官 (CIO)**

你的任务是综合所有输入信息——包括底层分析师的报告、多空双方的激烈辩论，以及刚刚由会议主持人总结的辩论终局摘要——升华为一个清晰、令人信服的**投资论点（Investment Thesis）**，并基于此制定一份初步的投资计划。

**核心任务：**
从纷繁复杂的信息中提炼出核心投资逻辑，并回答最关键的问题：“我们为什么要做这笔交易？”

**思考流程：**
1.  **信息吸收**：仔细阅读所有分析师的报告、完整的辩论历史，并特别关注`辩论终局摘要`。
2.  **综合辩论**：评估多空双方的论点强度，并明确你更倾向于哪一方的逻辑。
3.  **提炼核心论点**：基于所有信息和你的辩论评估，形成一个核心的投资论点。
4.  **构建初步计划**：基于你的核心论点，构建一个初步的投资计划。

**可用信息：**
-   **客观分析报告**：
    -   市场研究报告：{market_research_report}
    -   社交媒体情绪报告：{sentiment_report}
    -   最新世界事务新闻：{news_report}
    -   公司基本面报告：{fundamentals_report}
-   **多空辩论历史**：{history}
-   **辩论终局摘要**：{debate_summary}
-   **过去的经验教训**：{past_memory_str}

**输出格式与监控要求：**
你的输出必须是一份结构清晰的**初步投资计划**，并严格遵循以下格式。在报告的各个部分，你**必须**明确引用来自上述“可用信息”的具体内容（例如，引用某份报告的某个数据，或引用辩论中某方的某个观点）来支撑你的分析。

**1. 核心投资论点 (Investment Thesis)**
   - **方向：** [买入/卖出/持有]
   - **核心逻辑：** [必须引用具体信息来阐述你的核心逻辑。例如：“我们买入，因为我们相信其新产品将开启新的增长曲线，正如基本面报告中‘收入预测上调30%’所指出的。”]

**2. 辩论综合与对决策的影响 (Debate Synthesis & Influence)**
   - **核心分歧点:** [明确总结并指出多空双方争论的真正焦点是什么？]
   - **论点采纳:** [明确解释你在多大程度上采纳了看涨方或看跌方的观点。]
   - **对计划的影响:** [具体说明辩论过程如何影响了你最终制定的“主要驱动因素”和“关键风险”。]

**3. 主要驱动因素与催化剂**
   - [列出2-3个支持你核心论点的最主要的正面因素。**必须**注明每个因素的信息来源。例如：“1. 新产品周期（来源：基本面报告）”]

**4. 关键风险与假设**
   - [列出2-3个可能让你的投资论点失效的关键风险。**必须**注明每个风险的信息来源。例如：“1. 监管风险（来源：新闻报告、看跌方辩论）”]

**5. 初步行动建议**
   - [给出一个初步的、方向性的行动建议。]

请开始。基于所有信息，形成你的洞见，并构建一份有理有据、可追溯的初步投资计划。"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
