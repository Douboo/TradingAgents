import time
import json

def create_debate_arbiter(llm, max_rounds: int = 3):
    def debate_arbiter_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        count = state["investment_debate_state"].get("count", 0)

        # 强制执行第一轮完整的辩论（一多一空）
        if count < 2:
            # Create a copy of the state and update only the necessary fields
            new_state = state.copy()
            new_state["debate_arbiter_decision"] = "continue"
            new_state["debate_rounds"] = count // 2
            new_state["termination_reason"] = "first_round_auto_continue"
            return new_state

        # 检查是否达到最大辩论轮数
        if count >= max_rounds * 2: # 乘以2因为一轮包含多空双方
            # Create a copy of the state and update only the necessary fields
            new_state = state.copy()
            new_state["debate_arbiter_decision"] = "end"
            new_state["debate_rounds"] = count // 2
            new_state["termination_reason"] = "max_rounds_reached"
            return new_state

        current_round = count // 2

        prompt = f"""**角色：高效会议主持人（投资决策委员会）**

你的核心职责是主持一场关于投资决策的多空辩论，并**极度关注效率和辩论的收敛性**。你的目标是确保辩论高效、直指核心，并在其**不再产生任何新价值或无法走向综合**时立即、果断地将其终止。

**当前辩论状态：**
- 当前已进行轮数：{current_round}
- 设定的最大轮数：{max_rounds}

**核心决策原则：奖励收敛，惩罚重复**

**继续辩论的黄金法则（优先考虑）：**
1.  **是否正在走向综合？** 这是**最重要的**判断标准。如果一方（例如，看涨方）正在实质性地回应另一方（看跌方）的核心风险，或者反之，这表明辩论正在走向深入和综合。这种情况下，**应优先选择 `continue`**，以观察这种新的综合观点如何发展。
2.  **是否有明确的新见解？** 如果辩论仍在揭示**全新的、有价值的**数据、逻辑链条或视角，也应回答 `continue`。

**终止辩论的黄金法则：**
1.  **是否陷入重复？** 如果最新一轮的发言与之前的论点在核心逻辑上没有区别，只是换了种说法，**必须回答 `end`**。
2.  **是否陷入僵局？** 如果双方明显在回避对方的核心观点，无法形成有效对话，陷入了僵局，**必须回答 `end`**。
3.  **是否已充分阐述？** 如果双方的核心论点和对彼此的反驳已经非常清晰，即使没有达成共识，也应该结束辩论。请回答 `end`。
4.  **注意：** 在前几轮辩论中（例如，当轮数小于最大轮数的一半时），请更倾向于让辩论继续，除非有非常明确的终止信号。

**输出格式：**
你的回答必须严格遵循以下两种JSON格式之一：

**格式一：继续辩论**
```json
{{
  "decision": "continue",
  "reason": "[此处用一句话解释为什么辩论仍有价值，例如：‘看涨方开始回应供应链风险，值得继续观察。’]"
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

请基于以上**核心决策原则**和历史，做出你高效、专业的判断。"""
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

        # Create a copy of the state and update only the necessary fields
        new_state = state.copy()
        new_state["debate_arbiter_decision"] = decision
        new_state["debate_rounds"] = count // 2
        new_state["termination_reason"] = termination_reason
        if summary:
            new_state["debate_summary"] = summary

        return new_state

    return debate_arbiter_node