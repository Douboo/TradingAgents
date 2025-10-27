import json

def create_risk_arbiter(llm, max_rounds: int = 3):
    def risk_arbiter_node(state) -> dict:
        history = state["risk_debate_state"].get("history", "")
        count = state["risk_debate_state"].get("count", 0)

        # Force the first full round of debate (Risky, Safe, Neutral)
        if count < 3:
            # Create a copy of the state and update only the necessary fields
            new_state = state.copy()
            new_state["risk_arbiter_decision"] = "continue"
            new_state["risk_debate_rounds"] = count // 3
            new_state["termination_reason"] = "first_round_auto_continue"
            return new_state

        # Check if max rounds are reached
        if count >= max_rounds * 3:
            # Create a copy of the state and update only the necessary fields
            new_state = state.copy()
            new_state["risk_arbiter_decision"] = "end"
            new_state["risk_debate_rounds"] = count // 3
            new_state["termination_reason"] = "max_rounds_reached"
            return new_state

        prompt = f"""**角色：高效会议主持人（风险管理委员会）**

你的核心职责是主持一场关于投资计划的风险辩论，并**极度关注效率，绝不浪费决策者的时间**。你的目标是确保辩论富有成效，并在其**不再产生任何新价值**时立即、果断地将其终止。

**终止辩论的黄金法则（优先检查）：**
1.  **是否陷入重复？** 如果最新一轮的发言与之前的论点在核心逻辑上没有区别，只是换了种说法，**必须回答 `end`**。
2.  **是否无法取得进展？** 如果分析师们明显无法说服对方，陷入了僵局，**必须回答 `end`**。
3.  **是否已充分探讨？** 如果风险与回报之间的核心权衡已经被彻底探讨，即使没有达成共识，也应结束辩论。请回答 `end`。

**继续辩论的唯一条件：**
- **是否有明确的新见解？** 只有当辩论仍在揭示**全新的、有价值的**视角、风险量化或可行的优化方案时，才回答 `continue`。

**输出格式：**
你的回答必须严格遵循以下两种JSON格式之一：

**格式一：继续辩论**
```json
{{
  "decision": "continue",
  "reason": "[此处用一句话解释为什么辩论仍有价值，例如：‘保守派分析师提出了一个新的关于流动性风险的观点，需要得到回应。’]"
}}
```

**格式二：结束辩论并提供摘要**
```json
{{
  "decision": "end",
  "reason": "[此处用一句话解释为什么辩论应该结束，例如：‘辩论已陷入重复，最新观点未提供增量信息。’ 或 ‘关于仓位规模的核心分歧已经明确，进一步的辩论不太可能解决它。’]",
  "summary": {{
    "core_trade_off": "[总结正在讨论的核心权衡点，例如：‘激进的市场进入策略 vs. 资本保值策略。’]",
    "risky_proposal": "[总结激进派分析师最终的、最有说服力的建议。]",
    "safe_proposal": "[总结保守派分析师最终的、最有说服力的建议。]",
    "neutral_proposal": "[总结中立派分析师最终的、平衡的/量化的建议。]",
    "concluding_remark": "[给投资组合经理的一句结论性提醒，例如：‘关键的决策点在于，权衡中立派方案中可量化的上行空间与保守派所强调的尾部风险。’]"
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
            decision = "continue" # Default to continue on parsing failure
            termination_reason = "json_decode_error"
            summary = None

        # Create a copy of the state and update only the necessary fields
        new_state = state.copy()
        new_state["risk_arbiter_decision"] = decision
        new_state["risk_debate_rounds"] = count // 3
        new_state["termination_reason"] = termination_reason
        if summary:
            new_state["risk_debate_summary"] = summary

        return new_state

    return risk_arbiter_node