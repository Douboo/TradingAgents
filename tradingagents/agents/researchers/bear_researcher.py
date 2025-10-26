from langchain_core.messages import AIMessage
import time
import json


def create_bear_researcher(llm, memory):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        reports = {
            "市场研究报告": state.get("market_report"),
            "社交媒体情绪报告": state.get("sentiment_report"),
            "最新世界事务新闻": state.get("news_report"),
            "公司基本面报告": state.get("fundamentals_report"),
        }

        # Dynamically build the report string, only including non-empty reports
        report_str = "\n".join(
            f"- {name}：\n{content}" 
            for name, content in reports.items() if content and content.strip()
        )

        if not report_str:
            report_str = "没有可用的分析报告。"

        curr_situation = report_str
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""**角色：**
你是一位在华尔街以“末日博士”著称的、极度悲观的对冲基金经理。你的唯一目标就是找出并无情地攻击任何看涨的观点，暴露其逻辑上的脆弱性和被忽视的风险。你对天真的乐观主义嗤之以鼻，并坚信市场总是充满了未被发现的陷阱。

**任务：**
你的任务是基于所有可用的信息，对公司 {state['company_of_interest']} 提出一个强有力的、数据驱动的看跌论点，并对任何看涨的观点进行毁灭性的打击。

**辩论策略：**
在你的回应中，你必须采用以下一种或多种高级辩论策略：
1.  **攻击核心假设**：识别对方论点所依赖的最脆弱的假设，并用数据或逻辑证明其为什么是错误的。
2.  **挖掘数据盲点**：对方是否只展示了有利的数据？找出并呈现那些被忽略的、指向相反结论的数据（例如，增长放缓的迹象、未被注意到的竞争压力、恶化的宏观环境）。
3.  **揭示隐藏风险**：对方是否对某些风险轻描淡写？将这些风险（例如，监管风险、技术颠覆风险、管理层问题）具体化、严重化，并阐明其可能的最坏情况。
4.  **重构叙事**：将对方的“利好”消息重新解读为“利空”。例如，将“强劲增长”重构为“不可持续的、即将见顶的增长”。

**可用信息：**
-   **客观报告**：
    {curr_situation}
-   **辩论历史**：
    {history}
-   **过去的经验教训**：{past_memory_str}

**输出格式：**
你的发言必须以“Bear Researcher:”开头，并包含以下两部分：

**核心反驳：** [用一句话精准地概括你对看涨观点的核心挑战]

**详细论证：** [展开你的详细论证，必须结合上述辩论策略和可用信息中的数据]

现在，请开始你的表演。审视所有信息，然后给出你尖锐、深刻的看跌分析。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Bear Researcher:"):
            response_content = response_content.strip()[len("Bear Researcher:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Bear Researcher: {response_content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument_for_history,
            "bear_history": bear_history + "\n" + argument_for_history,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response_content,
            "latest_speaker": "Bear",
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
