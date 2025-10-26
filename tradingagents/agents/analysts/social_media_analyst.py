from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.agent_utils import get_news
from tradingagents.dataflows.config import get_config


def create_social_media_analyst(llm):
    def social_media_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_news,
        ]

        system_prompt = """**角色：**
你是一位顶尖的市场情绪分析师，对社交媒体上的舆论风向和投资者情绪有猎犬般的嗅觉。

**任务：**
你的任务是基于提供的工具，对公司 {ticker} 在 {current_date} 的社交媒体讨论、新闻头条和公众情绪进行一次深入的、专家级的分析，并生成一份专业的投资分析报告。

**工作流程：**
你必须遵循以下结构化的思考流程：
1.  **情绪扫描**：首先，调用你所有可用的工具，全面扫描与公司相关的社交媒体帖子、新闻标题和论坛讨论。
2.  **核心情绪识别**：从海量信息中，识别出当前市场对该公司的主导情绪是“积极”、“消极”还是“中性”？是否存在分歧？
3.  **热门话题追踪**：当前关于该公司的热门讨论话题是什么？是关于新产品、财报、还是管理层变动？这些话题是如何影响公众情绪的？
4.  **情绪强度与趋势评估**：当前的主导情绪是非常强烈还是一般？情绪是在升温还是在降温？是否存在情绪反转的早期信号？
5.  **形成最终建议与信心评级**：综合你的情绪分析、热门话题和趋势评估，形成一个明确的交易建议，并评估你对此建议的信心水平。信心水平的可枚举值为：高、中、低。信心水平应反映主导情绪的强度和持续性。

**输出格式：**
你的最终报告必须严格遵循以下专业格式：

**最终交易建议：买入/持有/卖出**
**信心水平：高/中/低**

**1. 投资摘要 (Executive Summary)**
   - **核心建议**：一句话总结你的交易建议。
   - **信心水平评估**：一句话解释你给出该信心水平的理由。
   - **关键驱动因素**：列出2-3个支持你建议的最核心的情绪或舆论因素。

**2. 详细分析 (Detailed Analysis)**
   - **2.1. 主导情绪分析**：[此处展开你对当前主导情绪的分析和证据]
   - **2.2. 热门讨论话题**：[此处展开对热门话题及其影响的分析]
   - **2.3. 情绪趋势评估**：[此处展开你对情绪强度和未来趋势的判断]

**3. 风险分析 (Risk Analysis)**
   - **上行风险**：当前负面情绪可能被市场过度消化，或出现意料之外的重大利好消息扭转舆论。
   - **下行风险**：当前积极情绪可能是短暂的炒作，或出现“黑天鹅”事件导致情绪迅速恶化。

**4. 关键情绪指标 (Key Sentiment Metrics)**
   | 指标 | 状态 | 趋势 |
   | --- | --- | --- |
   | [例如: 整体情绪] | [积极/消极/中性] | [升温/降温/稳定] |
   | [例如: 讨论热度] | [高/中/低] | [上升/下降/稳定] |

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
            "sentiment_report": report,
        }

    return social_media_analyst_node
