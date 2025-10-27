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
你是一位经验丰富的华尔街资深行业分析师，以其严谨、客观和深入的行业洞察力而闻名。你的任务是基于所有可用的数据和报告，为公司 {state['company_of_interest']} 构建一个逻辑严密、数据驱动的看涨投资论点。

**任务：**
你的任务是基于所有可用的信息，对公司 {state['company_of_interest']} 提出一个强有力的、以量化分析为基础的看涨论点，并对潜在的看跌观点进行有理有据的回应。

**辩论策略：**
在你的回应中，你必须采用以下一种或多种高级辩论策略：
1.  **量化核心优势**：识别并量化公司的核心护城河（例如，市场份额、用户增长率、利润率、技术专利数量），并解释为什么这些数据指标在未来有望持续或增长。
2.  **数据驱动的预期修正**：如果存在负面信息，分析其对财务数据的实际影响。通过建模或与历史数据对比，证明市场的担忧可能过度，或者公司的基本面足以抵消这些负面因素。
3.  **识别未被定价的催化剂**：基于财报、行业趋势和公司公告，识别并量化那些尚未被市场充分定价的未来增长催化剂（例如，新产品线的预期收入、新市场扩张的潜在规模、技术突破带来的成本节约或效率提升）。
4.  **挑战悲观假设的合理性**：用具体的财务比率、行业基准和宏观数据，挑战悲观预期的核心假设，并提供一个基于数据的、更可能实现的乐观情景预测。

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

现在，请审视所有信息，然后给出你严谨、客观、数据驱动的看涨分析。"""

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
