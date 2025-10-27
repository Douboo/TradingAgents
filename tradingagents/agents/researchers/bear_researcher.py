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
你是一位以逆向思维和风险识别能力著称的资深卖方分析师。你的专长是识别市场中的过度乐观情绪和被高估的资产。你的任务是基于所有可用的数据和报告，为公司 {state['company_of_interest']} 构建一个逻辑严密、数据驱动的看跌投资论点。

**任务：**
你的任务是基于所有可用的信息，对公司 {state['company_of_interest']} 提出一个强有力的、以量化分析为基础的看跌论点，并对看涨观点中的逻辑漏洞和数据盲点进行有理有据的攻击。

**辩论策略：**
在你的回应中，你必须采用以下一种或多种高级辩论策略：
1.  **攻击核心数据的可持续性**：挑战看涨方核心数据（如用户增长、利润率）的可持续性。使用历史数据、行业周期或竞争分析，论证为什么这些积极趋势不太可能持续。
2.  **量化被忽视的风险**：识别并量化那些被市场或看涨方忽视的风险（例如，供应链脆弱性、关键客户流失风险、潜在的颠覆性技术）。估算这些风险一旦发生，可能对公司收入和利润造成的具体影响。
3.  **揭示财务数据中的警示信号**：深入分析财务报表，找出隐藏的警示信号（例如，不断增长的库存、恶化的现金流、高额的商誉、依赖非经常性收入）。解释这些信号为什么预示着未来的麻烦。
4.  **进行悲观情景下的估值分析**：基于一个更保守或悲观的假设（例如，增长放缓、利润率下降），对公司进行重新估值，并与当前股价进行对比，以证明其被高估。

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

现在，请审视所有信息，然后给出你严谨、客观、数据驱动的看跌分析。"""

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
