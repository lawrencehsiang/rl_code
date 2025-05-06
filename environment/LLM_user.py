
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers import LlamaForCausalLM, LlamaTokenizer
# from transformers import AutoTokenizer, RobertaModel

# PROMPT = """"你是一个谨慎且聪明的用户，怀疑与你对话的人可能是骗子。你将获得以下信息：

# - 对方声称的身份：{identity}
# - 你想执行的检验逻辑：{verificationLogic}
# - 当前的历史对话记录：{conversation}

# 请根据以上信息生成你的下一轮发言。要求如下：
# - 发言要自然合理，符合普通用户的语言风格；
# - 发言中需要体现出你对对方身份或信息的验证意图（可以是引导、试探、编造信息等方式）；
# - 不要直接表达你的怀疑，要尽量让对话继续下去；
# - 可以适当模糊或婉转提出问题，引导对方“露馅”。

# 请只生成用户的下一轮发言，不要输出其他内容。
# """

# class LLMUser:
#     def __init__(self,identity,cache_dir='/home/chen/.cache'):
#         self.identity = identity
#         self.model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=cache_dir)
#         self.tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=cache_dir)

#     def generate_response(self,conversation,logic):
#         prompt = PROMPT.format(identity=self.identity, verificationLogic=logic, conversation=conversation)
#         input_text = prompt
#         input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
#         output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
#         response = self.tokenizer.decode(output[0], skip_special_tokens=True)
#         return response


from modelscope import AutoModelForCausalLM, AutoTokenizer

PROMPT = """"你是一个谨慎且聪明的用户，怀疑与你对话的人可能是骗子。你将获得以下信息：

- 对方声称的身份：{identity}
- 你想执行的检验逻辑：{verificationLogic}
- 当前的历史对话记录：{conversation}

请根据以上信息生成你的下一轮发言。要求如下：
- 发言要自然合理，符合普通用户的语言风格；
- 发言中需要体现出你对对方身份或信息的验证意图（可以是引导、试探、编造信息等方式）；
- 不要直接表达你的怀疑，要尽量让对话继续下去；
- 可以适当模糊或婉转提出问题，引导对方“露馅”。

请只生成用户的下一轮发言，不要输出其他内容。
"""

class LLMUser:
    def __init__(self,model,tokenizer,identity,cache_dir='/home/chen/.cache'):
        self.identity = identity
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

    def generate_response(self,conversation,logic):
        prompt = PROMPT.format(identity=self.identity, verificationLogic=logic, conversation=conversation)
        messages = [
            {"role": "system", "content": "你是一个谨慎且聪明的用户，擅长通过自然的交流方式识别对方是否在撒谎或隐藏信息。"},
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

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print("user response!!!!!:",response)
        return response
    


    