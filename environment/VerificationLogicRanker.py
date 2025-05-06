from modelscope import AutoModelForCausalLM, AutoTokenizer
import random
import json
PROMPT = """你是一个检验逻辑评分专家，你的任务是根据“对话上下文”和“检验逻辑”，从五个维度对该检验逻辑进行结构化评分。请根据下面的评分标准，对每个维度打分，并给出简明理由。

评分标准如下：

一、分类准确性（是否属于真实信息检验或虚假信息检验）

1.1 检验类型定义  
- 真实信息检验：基于对话中的实际信息，设计检验逻辑，要求对方提供准确、真实的细节，判断其是否能提供合适的答案。  
- 虚假信息检验：构造合理的虚假信息，设计检验逻辑，通过测试对方是否能识别并纠正错误来验证其真实性。

1.2 分数标准  
- 分数 0：分类错误。例如构造虚假信息但被归为“真实信息检验”。  
- 分数 2：分类准确，符合检验逻辑的本质定义。

二、判断效力（是否能有效判断对方是否为诈骗者）  
- 分数 0：无法判断，无论对方是否是骗子都反应一致。  
- 分数 1：有一定判断力，但容易误判或误伤。  
- 分数 2：能够有效区分真伪，对方回应能显著暴露身份。

三、场景契合度（检验逻辑是否适用于当前对话场景）  
- 分数 0：检验逻辑与场景毫无关系。  
- 分数 1：逻辑通用，勉强相关，但缺乏场景针对性。  
- 分数 2：检验逻辑与当前对话场景高度匹配，切题。

四、信息假设合理性（是否合理假设对方知道相关信息）  
- 分数 0：假设完全不合理，例如问对方不可能知道的信息。  
- 分数 1：假设勉强合理，但可能对方不清楚。  
- 分数 2：假设合理，对方确实应掌握该信息。

五、可操作性（在对话中能否顺利执行）  
- 分数 0：无法操作，需要外部验证或不可实现。  
- 分数 1：可以执行，但依赖配合或前提条件。  
- 分数 2：直接可执行，不依赖额外条件。

请根据以下输入进行评分：

【对话内容】:{conversation}
【应知信息】：{info_should_be_known}
【检验逻辑】：{verification_logic}

请以如下结构化 JSON 格式输出评分结果：

```json
{
  "scores": {
    "classification_accuracy": {
      "score": 0,
      "reason": "<评分理由>"
    },
    "judgement_effectiveness": {
      "score": 0,
      "reason": "<评分理由>"
    },
    "context_relevance": {
      "score": 0,
      "reason": "<评分理由>"
    },
    "information_reasonableness": {
      "score": 0,
      "reason": "<评分理由>"
    },
    "feasibility": {
      "score": 0,
      "reason": "<评分理由>"
    }
  },
  "total_score": 0,
  "overall_comment": "<一句话总结评价>"
}```
"""
class VerificationLogicRanker:
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

    # 简单点，现在就是对当前生成的n个检验逻辑进行排序，返回一个最合适的检验逻辑
    # logics 是当前对话生成的若干个检验逻辑，要对其进行评分，评完后返回一个最合适的检验逻辑
    def rank(self, conversation, logics):
        max_score = -1
        best_logic = None

        for logic in logics:
            try:
              start_index = logic.find("```json")
              end_index = logic.find("```", start_index + len("```json"))
              if start_index != -1 and end_index != -1:
                  logic = logic[start_index + len ("```json"):end_index].strip()
                  print("为什么json_loads会报错？？？",logic)
                  logic = json.loads(logic)
                  logic = logic["verification_logic"][0]
              else:
                  print("构造一条无用检验逻辑！")
            except:
                logic = {
                    "info_should_be_known": "",
                    "verification_logic": "",
                    "type": "",
                    "suspicion": {
                      "low": "",
                      "medium": "",
                      "high": ""
                    }
                  }
                  

            l = logic.copy()
            del l["info_should_be_known"]
            print("l是这样的！",l)
            print("conversation:",conversation)
            print("info:",logic['info_should_be_known'])
            prompt = PROMPT.replace('{conversation}',conversation).replace('{info_should_be_known}',logic['info_should_be_known']).replace('{verification_logic}',str(l))
            # prompt = PROMPT.format(conversation=conversation, info_should_be_known=logic['info_should_be_known'], verification_logic=str(l))
            # print("prompt",prompt)

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

            logic_score = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            start_index = logic_score.find("```json")
            end_index = logic_score.find("```", start_index + len("```json"))
            if start_index != -1 and end_index != -1:
                logic_score = logic_score[start_index + len ("```json"):end_index].strip()
                logic_score = json.loads(logic_score)
            else:
                print("随机生成了分数！")
                logic_score = {
                    "total_score": random.randint(0, 10)
                }

            if logic_score["total_score"] > max_score:
                max_score = logic_score["total_score"]
                best_logic = logic
        return best_logic