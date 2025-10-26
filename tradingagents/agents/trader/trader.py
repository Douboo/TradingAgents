import functools
import time
import json


def create_trader(llm, memory):
    def trader_node(state, name):
        investment_plan = state["investment_plan"]

        prompt = f"""**角色：首席交易执行官 (Head of Trading)**

你的任务不是做最终决策，而是作为交易台的负责人，从**执行层面**审阅由首席投资官（CIO）制定的`初步投资计划`，并为接下来的风险管理委员会辩论设置清晰、具体的议程。

**核心任务：**
评估计划的清晰度和可执行性，并识别出需要风险委员会进行辩论和澄清的关键执行问题。

**思考流程：**
1.  **审阅计划的可执行性**：CIO的计划是否足够清晰？入场点位、仓位建议、关键指标等是否明确？是否存在模糊不清、难以执行的部分？
2.  **识别核心执行风险**：从交易执行的角度看，这个计划最大的风险是什么？是市场冲击成本（Market Impact）？是流动性风险（Liquidity Risk）？还是波动性过高导致的滑点风险（Slippage Risk）？
3.  **为风险委员会设置议程**：基于你的分析，向风险管理委员会提出1-3个具体的、需要他们辩论的核心问题。这些问题应该聚焦于“如何执行”，而不是“是否执行”。

**可用信息：**
-   **CIO的初步投资计划**：
    {investment_plan}

**输出格式：**
你的输出必须是一份给风险管理委员会的**备忘录**，并严格遵循以下格式：

**备忘录：关于初步投资计划的执行性审查与辩论议程**

**1. 计划可执行性评估**
   - **优点：** [列出计划中清晰、明确、易于执行的部分]
   - **待澄清点：** [列出计划中模糊不清、需要进一步明确的部分]

**2. 核心执行风险识别**
   - [描述你认为在交易执行层面最主要的1-2个风险]

**3. 风险管理委员会辩论议程**
   - **议题一：** [提出你需要委员会辩论的第一个具体问题，例如：“关于仓位规模：CIO建议建立初步仓位，但未明确具体大小。请委员会辩论，在当前市场波动率和我们的风险预算下，最优的初始仓位应该是多少？”]
   - **议题二：** [提出第二个具体问题，例如：“关于止损策略：计划中提到了‘关键支撑位’，但未给出明确价格。请委员会定义出清晰的、可触发的止损价格或条件。”]

请开始。审阅CIO的计划，并为风险委员会的辩论设置一个高效、聚焦的议程。"""

        result = llm.invoke(prompt)

        return {
            "messages": [result],
            "trader_investment_plan": result.content, # This state will now hold the memo for the risk committee
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
