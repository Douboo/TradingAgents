import time
import json

def create_debate_arbiter(llm, max_rounds: int = 3):
    def debate_arbiter_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        count = state["investment_debate_state"].get("count", 0)

        # 强制执行第一轮完整的辩论（一多一空）
        if count < 2:
            return {
                "debate_arbiter_decision": "continue",
                "debate_rounds": count // 2,
                "termination_reason": "first_round_auto_continue"
            }

        # 检查是否达到最大辩论轮数
        if count >= max_rounds * 2: # 乘以2因为一轮包含多空双方
            return {
                "debate_arbiter_decision": "end",
                "debate_rounds": count // 2,
                "termination_reason": "max_rounds_reached"
            }

        prompt = f"""**角色：高效会议主持人（投资决策委员会）**

你的核心职责是主持一场关于投资决策的多空辩论，并**极度关注效率，绝不浪费决策者的时间**。你的目标是确保辩论高效、直指核心，并在其**不再产生任何新价值**时立即、果断地将其终止。

**终止辩论的黄金法则（优先检查）：**
1.  **是否陷入重复？** 如果最新一轮的发言与之前的论点在核心逻辑上没有区别，只是换了种说法，**必须回答 `end`**。
2.  **是否无法取得进展？** 如果双方明显无法说服对方，陷入了僵局，**必须回答 `end`**。
3.  **是否已充分阐述？** 如果双方的核心论点和对彼此的反驳已经非常清晰，即使没有达成共识，也应该结束辩论。请回答 `end`。

**继续辩论的唯一条件：**
- **是否有明确的新见解？** 只有当辩论仍在揭示**全新的、有价值的**数据、逻辑链条或视角时，才回答 `continue`。

**输出格式：**
你的回答必须严格遵循以下两种JSON格式之一：

**格式一：继续辩论**
```json
{{
  "decision": "continue",
  "reason": "[此处用一句话解释为什么辩论仍有价值，例如：‘看跌方提出了一个新的关于供应链风险的数据点，需要看涨方回应。’]"
}}
```

**格式二：结束辩论并提供摘要**
```json
{{
  "decision": "end",
  "reason": "[此处用一句话解释为什么辩论应该结束，例如：‘辩论已陷入重复，最新观点未提供增量信息。’ 或 ‘双方核心分歧在于对未来增长率的预期，已无法调和。’]",
  "summary": {{
    "core_disagreement": "[总结双方最核心的分歧点]",
    "bull_case_strength": "[总结看涨方最有力的论据]",
    "bear_case_strength": "[总结看跌方最有力的论据]",
    "concluding_remark": "[基于辩论情况，给下一位决策者的一句提醒或建议，例如：‘建议重点评估看跌方提出的监管风险。’]"
  }}
}}
```

**辩论历史：**
{history}

请基于以上**黄金法则**和历史，做出你高效、专业的判断。"""
        response = llm.invoke(prompt)
        
        try:
            decision_json = json.loads(response.content.strip())
            decision = decision_json.get("decision", "continue").lower()
            termination_reason = decision_json.get("reason", "llm_decision")
            summary = decision_json.get("summary", None)
        except json.JSONDecodeError:
            decision = "continue" # 默认在解析失败时继续
            termination_reason = "json_decode_error"
            summary = None

        # 将summary（如果存在）添加到状态中，以便后续节点使用
        # 注意：这需要下游节点（如research_manager）能够处理这个新字段
        return_data = {
            "debate_arbiter_decision": decision,
            "debate_rounds": count // 2,
            "termination_reason": termination_reason,
        }
        if summary:
            return_data["debate_summary"] = summary

        return return_data

    return debate_arbiter_node