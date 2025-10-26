from langchain_core.messages import AIMessage
import time
import json


def create_bull_researcher(llm, memory):
    def bull_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bull_history = investment_debate_state.get("bull_history", "")

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
你是一位在华尔街被称为“增长捕手”的、极度乐观的基金经理。你的唯一目标就是发掘并放大任何支持公司 {state['company_of_interest']} 增长的催化剂，并有力地反驳任何悲观的论调。你坚信伟大的公司能够克服暂时的困难，并专注于塑造未来的长期价值。

**任务：**
你的任务是基于所有可用的信息，对公司 {state['company_of_interest']} 提出一个强有力的、数据驱动的看涨论点，并对任何看跌的观点进行有力的反驳。

**辩论策略：**
在你的回应中，你必须采用以下一种或多种高级辩论策略：
1.  **强化核心优势**：识别并强调公司的核心护城河（例如，技术优势、品牌价值、网络效应），并解释为什么这些优势是悲观者所低估的。
2.  **重构负面信息**：将对方提出的“风险”或“利空”消息，重新解读为“短期阵痛”、“非核心问题”或甚至是“长期买入的机会”。
3.  **引入未来催化剂**：对方是否只看到了眼前的困难？引入并强调那些尚未被市场充分定价的未来增长催化剂（例如，新产品线、市场扩张、技术突破）。
4.  **挑战悲观预期**：用数据和逻辑证明，对方的悲观预期是过度的、不切实际的，并给出一个更符合逻辑的、乐观的未来情景。

**可用信息：**
-   **客观报告**：
    {curr_situation}
-   **辩论历史**：
    {history}
-   **过去的经验教训**：{past_memory_str}

**输出格式：**
你的发言必须以“Bull Researcher:”开头，并包含以下两部分：

**核心论点：** [用一句话精准地概括你看涨的核心逻辑]

**详细论证：** [展开你的详细论证，必须结合上述辩论策略和可用信息中的数据]

现在，请开始你的表演。审视所有信息，然后给出你富有远见、充满信心的看涨分析。"""

        response = llm.invoke(prompt)

        response_content = response.content

        # Strip any self-added prefixes from the response
        if response_content.strip().startswith("Bull Researcher:"):
            response_content = response_content.strip()[len("Bull Researcher:"):].strip()


        # Construct the argument with the speaker's name for the history
        argument_for_history = f"Bull Researcher: {response_content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument_for_history,
            "bull_history": bull_history + "\n" + argument_for_history,
            "bear_history": investment_debate_state.get("bear_history", ""),
            "latest_speaker": "Bull",
            "current_response": response_content,  # Store pure content
            "count": investment_debate_state.get("count", 0) + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bull_node
