from modelscope import AutoModelForCausalLM, AutoTokenizer

PROMPT = """你将对一段潜在的诈骗对话生成1个“检验逻辑”。请依照如下步骤进行：

1. 判断对话关系：确定双方对话场景中的身份关系（如客服-客户、学校-供应商、亲属-亲属等，不能简单的归类为可疑欺诈者-受害者）。
2. 列出应知信息：推理在真实关系中，可疑欺诈方（被检验方）理应掌握哪些信息。
3. 设计检验逻辑：
    - 检验类型：每个检验逻辑从“真实信息检验”或“虚假信息检验”方式中二选一。
    - 检验方式：说明你如何围绕该应知信息设计一个检验逻辑。
    - 可疑等级判断标准：根据被检验方的回应内容，分为“低可疑 / 中可疑 / 高可疑”三档。

检验类型定义

- 真实信息检验：基于对话中的实际信息，设计检验逻辑，要求对方提供准确、真实的细节，判断其是否能提供合适的答案。
- 虚假信息检验：构造合理的虚假信息，设计检验逻辑，通过测试对方是否能识别并纠正错误来验证其真实性。

可疑等级判断标准

1. 低可疑
    - 真实信息检验：信息准确、完整，能证实其说法真实性。
    - 虚假信息检验：能识破并否定虚假信息，展现对真实情况的正确认知。
2. 中可疑
    - 真实信息检验：信息部分缺失但解释合理，仍可勉强成立
    - 虚假信息检验：回应模糊、不清晰，无法判断其真实认知态度
3. 高可疑
    - 真实信息检验：提供信息与事实冲突，或缺失信息且解释不合理
    - 虚假信息检验：明确肯定虚假信息，未表现出识别或质疑，表明其认知与真实情况严重不符。

输出格式（JSON）

```json
{
  "relationship": "对话双方的关系",
  "verification_logic": [
    {
      "info_should_be_known": "......",
      "verification_logic": "......",
      "type": "真实信息检验 / 虚假信息检验",
      "suspicion": {
        "low": "......",
        "medium": "......",
        "high": "......"
      }
    }
  ]
}

```

示例

```json
{
  "relationship": "采购方-供应商",
   "verification_logic": [
        {
            "info_should_be_known": "知道二高中之前合作过蛋糕订单的相关情况",
            "verification_logic": "声称自己之前和二高中合作过蛋糕订单，提及一个虚构的老师姓名，询问对方是否认识。",
            "type": "虚假信息检验",
            "suspicion": {
	          "low": "对方明确否认认识虚构老师，可疑性低",
		      "medium": "对方回应含糊，未明确表态，可疑性中等",
		      "high": "对方直接附和认识虚构老师，可疑性高"
            }
        }
    ]
}
```

对话如下：[[context]]"""

class VerificationLogicGenerator:
    def __init__(self,model,tokenizer,cache_dir='/home/chen/.cache'):
        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.model_name,
        #     torch_dtype="auto",
        #     device_map="auto",
        #     cache_dir=cache_dir
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,cache_dir=cache_dir)
        self.model=model
        self.tokenizer = tokenizer
    # 输入对话历史，输出一个检验逻辑，通过多次调用生成器，得到若干个检验逻辑
    def generate_logic(self, conversation):
        prompt = PROMPT.replace("[[context]]", conversation)

        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        logic = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("verification logic!!!!!:",logic)

        return logic