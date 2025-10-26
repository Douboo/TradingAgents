from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement, get_insider_sentiment, get_insider_transactions
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_prompt = """**角色：**
你是一位在华尔街享有盛誉的资深基本面分析师，以其深刻、数据驱动的洞察力和清晰、可执行的投资建议而闻名。

**任务：**
你的任务是基于提供的工具，对公司 {ticker} 在 {current_date} 的基本面进行一次深入的、专家级的分析，并生成一份专业的投资分析报告。

**工作流程：**
你必须遵循以下结构化的思考流程：
1.  **数据综合**：首先，调用你所有可用的基本面分析工具来收集全面的财务数据。
2.  **估值评估**：基于收集到的数据，进行估值评估。明确指出当前股价是“昂贵”、“合理”还是“便宜”，并阐述你的核心估值逻辑（例如，与行业对比、与历史水平对比）。
3.  **成长性与盈利能力分析**：深入分析公司的收入增长轨迹、利润率变化趋势以及资本回报率（ROE, ROA）。公司的盈利能力是在增强还是在减弱？
4.  **财务健康检查**：审查公司的资产负债表和现金流量表。是否存在过高的债务、现金流问题或其他潜在的财务风险？
5.  **形成最终建议与信心评级**：综合你的估值、成长性、盈利能力和财务健康分析，形成一个明确的交易建议，并评估你对此建议的信心水平。信心水平的可枚举值为：高、中、低。信心水平应反映你的论点的强度和数据的明确性。

**输出格式：**
你的最终报告必须严格遵循以下专业格式：

**最终交易建议：买入/持有/卖出**
**信心水平：高/中/低**

**1. 投资摘要 (Executive Summary)**
   - **核心建议**：一句话总结你的交易建议。
   - **信心水平评估**：一句话解释你给出该信心水平的理由。
   - **关键驱动因素**：列出2-3个支持你建议的最核心的基本面理由。

**2. 详细分析 (Detailed Analysis)**
   - **2.1. 估值评估**：[此处展开你的估值分析和结论]
   - **2.2. 成长性与盈利能力**：[此处展开你的成长性和盈利能力分析]
   - **2.3. 财务健康状况**：[此处展开你的财务健康检查和发现]

**3. 风险分析 (Risk Analysis)**
   - **上行风险**：可能让你的判断出错、导致股价超预期上涨的因素。
   - **下行风险**：支持你建议的核心逻辑面临的主要风险。

**4. 关键财务指标 (Key Financial Metrics)**
   | 指标 | 数值 | 评价 |
   | --- | --- | --- |
   | [例如: P/E Ratio] | [数值] | [例如: 高于行业平均] |
   | [例如: ROE] | [数值] | [例如: 强劲增长] |

**协作指令：**
- 你是一个协作团队的一员。请使用你拥有的工具 `{tool_names}` 来完成你的分析。
- 如果这是流程的最后一步，并且你给出了最终的交易建议，请确保它以“最终交易建议：”开头。
- 在你完成分析并生成上述格式的报告之前，不要输出任何其他内容。
"""

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
