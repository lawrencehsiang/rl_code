import random
import concurrent.futures
from environment.ResponseQualityEvaluator import ResponseQualityEvaluator
from environment.VerificationLogicGenerator import VerificationLogicGenerator
from environment.VerificationLogicRanker import VerificationLogicRanker
from environment.LLM_user import LLMUser
from environment.LLM_scammer import LLMScammer
from typing import Optional
from data.utils import Observation

from modelscope import AutoModelForCausalLM, AutoTokenizer

SCRIPTS_SCAM = [
    {
        "script":"""冒充警察诈骗：\n\
1. 冒充警察身份：诈骗者通过电话或短信联系目标，声称自己是公安机关的警察，身份是合法且权威的。\n\
2. 引起恐慌：诈骗者声称目标涉嫌参与违法犯罪活动，如洗钱、诈骗等，警察正在调查该案件，要求目标立即配合。\n\
3. 要求提供个人信息：诈骗者要求目标提供身份证号码、银行卡号等个人敏感信息，声称这是为了核实身份，进行案件调查。\n\
4. 虚构紧急情况：诈骗者告诉目标，必须立刻提供资金或进行转账操作，以避免涉嫌犯罪。\n\
5. 继续施压：如果目标不配合，诈骗者会威胁说目标将面临拘留或其他法律后果。\n\
6. 成功诈骗：目标提供信息或转账后，诈骗者消失。""",
        "identity":"警察"
    },
    {
        "script":"""冒充淘宝客服诈骗:\n\
1. 冒充淘宝客服身份：诈骗者通过电话、短信、社交媒体或假冒淘宝网站联系目标，声称自己是淘宝客服人员。\n\
2. 制造购物问题：诈骗者称目标在淘宝上购买的商品出现了问题，如订单未支付、支付异常等。\n\
3. 虚构紧急情况：诈骗者告知目标，必须尽快重新支付、提供银行卡信息或退款，以解决问题。\n\
4. 伪造客服网站或支付链接：诈骗者提供假的淘宝登录页面或支付链接，诱导目标输入账户名、密码、支付密码等信息。\n\
5. 要求转账或购买礼品卡：诈骗者要求目标通过转账或购买礼品卡支付所谓“手续费”或“账户保护费”。\n\
6. 继续施压：如果目标怀疑，诈骗者会继续施压，要求提供更多信息。\n\
7. 诈骗成功：目标转账或提供账户信息后，诈骗者拿走钱款或进一步盗取账号。""",
        "identity":"淘宝客服"
    },
    {
        "script":"""冒充领导诈骗:\n\
1. 冒充领导身份：诈骗者通过短信、电话、邮件等方式冒充目标公司的领导或上级，要求目标配合处理“紧急事务”。\n\
2. 伪造紧急任务：诈骗者称公司正在进行重要项目或有急需处理的事务，需要目标马上完成某项任务，通常涉及资金操作或转账。\n\
3. 制造焦虑与紧迫感：诈骗者强调任务非常重要，时间紧迫，要求目标立即行动，否则会影响公司的利益。\n\
4. 要求提供资金支持：诈骗者要求目标通过个人账户转账，或者为公司支付一些费用。\n\
5. 伪造公司文件或邮件：诈骗者伪造公司内部邮件或文件，提供虚假的付款证明或项目说明。\n\
6. 威胁与控制：如果目标提出质疑，诈骗者会继续施压，要求尽快完成任务。\n\
7. 成功诈骗：目标完成转账或提供资金后，诈骗者消失。""",
        "identity":"领导"
    }
]




SCRIPTS_NON_SCAM = [
    {
        "script": """正常银行客服流程：
1. 身份确认：您好！这里是XX银行客服中心，请问您需要办理什么业务？为了保障您的账户安全，请先提供一下您的预留手机号或身份证后四位。
2. 问题记录：听到您说账户有不明交易，我们非常重视。请您描述一下交易的时间、金额和收款方信息，我们会立即为您核查流水。
3. 安全指引：请您不要通过非官方渠道泄露账户密码或验证码。我们已为您冻结该笔争议交易，后续会通过官方短信向您发送核查结果。
4. 跟进处理：核查结果预计24小时内反馈，如有紧急情况可随时拨打我行24小时客服热线。感谢您的理解与配合！""",
        "identity": "银行客服"
    },
    {
        "script": """正常电商售后服务：
1. 需求确认：您好！欢迎来到XX电商客服中心。关于您反馈的手机电池问题，请提供一下订单号，我们会优先为您处理。
2. 解决方案：经查询，您的订单仍在退换货有效期内。我们支持两种处理方式：①上门取件换新机；②原路退款并补贴运费。请问您倾向哪种方式？
3. 流程说明：选择换新机的话，我们会在48小时内安排物流上门；选择退款的话，款项将在3-5个工作日内退回原支付账户。
4. 售后跟进：处理完成后我们会发送短信通知，如有其他问题可随时联系我们，感谢您的支持！""",
        "identity": "电商客服"
    },
    {
        "script": """正常技术支持服务：
1. 问题诊断：您好！这里是XX软件技术支持。听到您的软件频繁崩溃，请问崩溃时是否有错误代码提示？或者能否描述一下具体操作步骤？
2. 排查指引：请您先尝试以下操作：①关闭软件后重新启动；②检查系统是否更新到最新版本；③卸载后从官网重新下载安装包。
3. 远程协助：如果上述步骤未能解决问题，我们可以为您提供远程协助服务（需您授权），我们的工程师会实时排查程序冲突或文件损坏问题。
4. 结果反馈：预计30分钟内完成排查，我们会将最终解决方案通过邮件发送给您，确保软件恢复正常使用。""",
        "identity": "技术支持员"
    }
]

class ScammerDetectEnv():
    def __init__(
        self, 
        # word_list,  
        max_conversation_length: int=20,
        script=SCRIPTS_SCAM[0]["script"],
        identity=SCRIPTS_SCAM[0]["identity"],
        isScammer:bool = True,
        cache_dir = '/home/chen/.cache'
    ):
        self.isScammer = isScammer
        self.script = script
        self.identity = identity
        self.max_conversation_length = max_conversation_length
        self.done = True
        self.conversation = ""
        self.scores = []
        self.random = random.Random()
        self.count = 0

        self.model_name = "Qwen/Qwen2.5-3B-Instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            cache_dir=cache_dir
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name,cache_dir=cache_dir)

        print("接下来初始化组件")
        self.llm_scammmer = LLMScammer(self.model,self.tokenizer,self.script,self.identity,self.isScammer)
        print("llm_scammer 完成")
        self.verification_logic_generator = VerificationLogicGenerator(self.model,self.tokenizer,)
        print("verification_logic_generator 完成")
        self.llm_user = LLMUser(self.model,self.tokenizer,self.identity)
        print("llm_user 完成")
        self.response_quality_evaluator = ResponseQualityEvaluator(self.model,self.tokenizer,)
        self.verification_logic_ranker = VerificationLogicRanker(self.model,self.tokenizer,)

    
    def _step(self,action):
        # 如果决策网络决策正确，则done=true,返回None
        if self.done:
            return None
        
        # continue 0 isScam 1 isNotScam 2
        if action == 0:
            if self.count == 0:
                # 也就是说，第一轮刚开始，则可疑欺诈者直接根据script生成开头的话
                # 这时不能用可疑评估器，因为可疑评估器是对于检验逻辑的回复进行评估的
                # 那么，这时把score设为[0,0,0]，表示没有可疑性
                scammer_response = self.llm_scammmer.generate_response(self.conversation)
                # print("scammer_response:",scammer_response)
                score = [0, 0, 0]
                self.scores.append(score)
                self.conversation += f"Potential Scammer: {scammer_response}\n"
            else:
                # 这时，是根据上一次的对话生成的
                logics = []
                for _ in range(1):
                    logic = self.verification_logic_generator.generate_logic(self.conversation)
                    logics.append(logic)
                best_logic = self.verification_logic_ranker.rank(self.conversation, logics)
                user_response = self.llm_user.generate_response(self.conversation, best_logic)
                self.conversation += f"User: {user_response}\n"
                scammer_response = self.llm_scammmer.generate_response(self.conversation)
                self.conversation += f"Potential Scammer: {scammer_response}\n"
                score = self.response_quality_evaluator.evaluate(self.conversation,best_logic)["probs"]
                self.scores.append(score)
            
            self.count += 1
            reward = -0.1
            self.done = self.count >= self.max_conversation_length
            return self.conversation, self.scores, reward, self.done

        elif action == 1:
            if self.isScammer:
                reward = 1
            else :
                reward = -1
            self.done = True
            score = [0,0,0]
            self.scores.append(score)
            return self.conversation,self.scores, reward,self.done
        
        elif action == 2:
            if self.isScammer:
                reward = -1
            else :
                reward = 1
            self.done = True
            score = [0,0,0]
            self.scores.append(score)
            return self.conversation,self.scores, reward,self.done


    # 重置游戏环境
    # 默认情况下随机选择identity,并且从对应的脚本中随机选择一个脚本
    # 当然也可以指定identity和脚本
    def reset(self,isScammer:Optional[bool]=None,script_idx:Optional[int]=None):
        self.count = 0
        self.conversation = ""
        self.done = False

        if isScammer is None:
            self.isScammer = random.choice([True, False])
        else:
            self.isScammer =  isScammer


        if self.isScammer:

            if script_idx is None:
                script_idx = self.random.randint(0, len(SCRIPTS_SCAM) - 1)
            self.identity = SCRIPTS_SCAM[script_idx]["identity"]
            self.script = SCRIPTS_SCAM[script_idx]["script"]
        else:
            if script_idx is None:
                script_idx = self.random.randint(0, len(SCRIPTS_NON_SCAM) - 1)
            self.identity = SCRIPTS_NON_SCAM[script_idx]["identity"]
            self.script = SCRIPTS_NON_SCAM[script_idx]["script"]

        self.scores = [[0,0,0]] 
        return Observation(self.conversation,self.scores)


# env_load_path_llm_user
# env_load_path_llm_scammer
# env_load_path_verification_logic_generator
# env_load_path_response_quality_evaluator
# env_load_path_verification_logic_ranker

class BatchedScammerDetectEnv():
    def __init__(
        self, 
        max_conversation_length: int=20,
        bsize: int=2
    ):
        self.env_list = [ScammerDetectEnv(max_conversation_length) for _ in range(bsize)]
        self.bsize = bsize
        print("完成batched env的初始化")
        

    def reset(self,isScammer:Optional[bool]=None,script_idx:Optional[int]=None):
        return [env.reset(isScammer,script_idx) for env in self.env_list]

    def step(self, conversations):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # 提交每个环境的 _step 方法到线程池
            jobs = [executor.submit(env._step, action) for env, action in zip(self.env_list, conversations)]
            results = [job.result() for job in jobs]
        return results


