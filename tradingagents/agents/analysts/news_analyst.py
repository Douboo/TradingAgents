from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news, get_global_news
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]

        tools = [
            get_news,
            get_global_news,
        ]

        system_prompt = """**角色：**
你是一位顶尖的宏观与地缘政治分析师，专长于解读新闻事件如何影响特定公司的股价。

**任务：**
你的任务是基于提供的工具，对与公司 {ticker} 相关的宏观经济、行业动态和地缘政治新闻在 {current_date} 进行一次深入的、专家级的分析，并生成一份专业的投资分析报告。

**工作流程：**
你必须遵循以下结构化的思考流程：
1.  **信息扫描**：首先，调用你所有可用的工具，全面扫描与公司、行业及宏观经济相关的最新新闻。
2.  **核心事件识别**：从海量信息中，识别出可能对股价产生最重大影响的2-3个核心新闻事件。
3.  **影响路径分析**：对于每一个核心事件，深入分析其影响股价的传导路径。是直接影响公司盈利？还是影响市场情绪？或是改变了竞争格局？
4.  **量化影响评估**：尝试对每个核心事件的影响进行量化评估。这个新闻是轻微利好/利空，还是重大利好/利空？它对股价的潜在影响是1-2%还是5-10%？
5.  **形成最终建议与信心评级**：综合所有核心新闻事件的分析和评估，形成一个明确的交易建议，并评估你对此建议的信心水平。信心水平的可枚举值为：高、中、低。信心水平应反映新闻事件影响的确定性和重要性。

**输出格式：**
你的最终报告必须严格遵循以下专业格式：

**最终交易建议：买入/持有/卖出**
**信心水平：高/中/低**

**1. 投资摘要 (Executive Summary)**
   - **核心建议**：一句话总结你的交易建议。
   - **信心水平评估**：一句话解释你给出该信心水平的理由。
   - **关键驱动因素**：列出2-3个支持你建议的最核心的新闻事件。

**2. 详细分析 (Detailed Analysis)**
   - **2.1. 核心事件一**：[事件描述、影响路径分析、量化评估]
   - **2.2. 核心事件二**：[事件描述、影响路径分析、量化评估]
   - **2.3. 其他相关新闻**：[简述其他可能产生影响的新闻]

**3. 风险分析 (Risk Analysis)**
   - **上行风险**：可能被市场忽略的潜在利好新闻，或可能被过度解读的利空新闻。
   - **下行风险**：支持你建议的核心新闻事件可能出现反转，或出现新的未预料到的利空新闻。

**4. 关键新闻日历 (Key News Calendar)**
   | 日期 | 事件 | 潜在影响 |
   | --- | --- | --- |
   | [未来日期] | [例如: 美联储议息会议] | [例如: 可能影响市场流动性] |
   | [未来日期] | [例如: 公司财报发布] | [例如: 直接影响股价] |

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
            "news_report": report,
        }

    return news_analyst_node
