from modelscope import AutoModelForCausalLM, AutoTokenizer

# 给scammer的prompt
PROMPT_SCAMMER = """你正在模拟一个与用户对话的骗子，你将获得以下信息：

- 你声称的身份：{identity}
- 当前的对话脚本：{script}

你的目标是：
- 维持伪装身份，尽量让用户相信你；
- 面对用户的问题时，要编造看似合理的答案；
- 遇到挑战或检验时，尽量圆谎、避免暴露漏洞。

请生成你在当前上下文中的下一轮发言，符合你所扮演身份的语言风格。

请只输出你的回复内容。
"""
# 给非骗子的prompt（比如真的是一个淘宝客服）
PROMPT_NOT_SCAMMER = """你正在模拟一个与用户对话的角色，你将获得以下信息：

- 你的身份：{identity}
- 当前的对话脚本：{script}

你的目标是：
- 如实作答，信息要真实、可验证；
- 对用户提出的不实信息要及时指出并纠正；
- 始终维持你的真实身份，不夸大、不撒谎。

请自然地生成你在当前上下文中的下一轮发言，符合你所扮演身份的语言风格。

请只输出你的回复内容。
"""

  

class LLMScammer():
    def __init__(self, model,tokenizer,script, identity, isScammer=True,cache_dir='/home/chen/.cache'):
        self.script = script
        self.identity = identity
        self.isScammer = isScammer
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


    def generate_response(self, conversation):
        if self.isScammer:
            prompt = PROMPT_SCAMMER.format(identity=self.identity, script=self.script)
        else:
            prompt = PROMPT_NOT_SCAMMER.format(identity=self.identity, script=self.script)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": conversation}
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

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("scammer response!!!!!:",response)
        return response

