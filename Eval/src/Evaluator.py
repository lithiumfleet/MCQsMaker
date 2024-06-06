from openai import AsyncOpenAI

from tqdm import tqdm
import asyncio
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from abc import abstractmethod, ABC
import re
import inspect


logger = logging.Logger(name="log")


class EvalDataset:
    def __init__(self) -> None:
        self.question_list:list[Question] = None

    def load_from_file(self, file_path:str|list[str], clear_last:bool) -> None:
        if isinstance(file_path, str):
            file_path = [file_path]

        for index, path in enumerate(file_path):
            if index == 0:
                if clear_last:
                    self._load_from_file(path, append_mode=False)
                else:
                    self._load_from_file(path, append_mode=True)
            else:
                self._load_from_file(path, append_mode=True)

        logger.info("Finish load all eval datasets.")


    def _load_from_file(self, file_path:str, append_mode:bool) -> None:
        try:
            with open(file_path, "r", encoding="u8") as fp:
                json_content = json.load(fp)
            assert isinstance(json_content, list)
            assert isinstance(json_content[0], dict)
        except Exception as err:
            logger.error(f"An exception occurred while loading json file from {file_path}")
            raise err
        
        if not append_mode or append_mode and self.question_list is None:
            self.question_list = []

        for question in json_content:
            # TODO: need a handler to check question type, not only MCQ
            question["id"] = str(self.size+1)
            self.question_list.append(MCQ(**question))
        self,logger.info(f"success fully load from {file_path}")
    
    @property
    def size(self) -> int:
        return len(self.question_list)


class Question(ABC):

    @abstractmethod
    def check(self, resp:str) -> bool:
        return NotImplementedError

    @abstractmethod
    def to_prompt(self) -> str:
        return NotImplementedError


@dataclass
class MCQ(Question):
    """
    This class includes multiple choices question & single choice question. Both will be handled correctly.
    """
    id:str
    question:str
    options:dict[str:str]
    answer:str
    analysis:str

    @classmethod
    def from_dict(cls, env):      
        return cls(**{
            k: v for k, v in env.items() if k in inspect.signature(cls).parameters
        })

    def check(self, resp:str) -> bool:
        try:
            resp_answer:str = re.findall(r"[ABCD]{1,4}", resp)[-1]
        except Exception as err:
            logger.warning(f'No answer was found in response. Automatically return False. MCQ:{self.question[:10]+"..."+self.question[-10:]}\t=>\tResp:{resp[:10]+"..."+resp[-10:]}')
            return False
        return set(resp_answer) == set(self.answer)

    def to_prompt(self) -> str:
        qtype = "单选题" if len(self.answer) == 1 else "多选题"
        str_options = "\n".join([f"{k}.{v}" for k, v in self.options.items()])
        system_prompt = f"你是一位专业的心理咨询助手，拥有丰富的心理学知识。现在有一道心理学知识的{qtype}，需要你利用自己的心理学知识进行解答。\n"
        instruction = f"**题目**：\n{self.question}\n**选项**：\n{str_options}\n\n\n请先给出你的解题思路，再从ABCD四个选项中选出你认为正确的选项。你的答案是："
        # TODO: not support system prompt yet.
        return system_prompt+instruction
    
    
class AsyncClient:
    def __init__(self, base_url, model_name:str, api_key) -> None:
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    async def chat(self,prompt:str|list[str]) -> str:
        if isinstance(prompt, str):
            messages = [ {"role": "user", "content": prompt} ]
        else:
            messages = []
            for i, p in enumerate(prompt):
                if i % 2 == 0:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        return response.choices[0].message.content
    
    async def _batch_chat(self, prompts:list[str|list[str]]) -> list[str]:
        assert isinstance(prompts, list)
        return await asyncio.gather(*[self.chat(prompt) for prompt in prompts])

    async def test_question(self, question:Question) -> bool|tuple[Question,str]:
        resp:str =  await self.chat(question.to_prompt())
        return question.check(resp), (question,resp)

    async def answer_question(self, question:Question) -> tuple[Question,str]:
        """this function will return resp and question"""
        resp:str =  await self.chat(question.to_prompt())
        return (question,resp)
    

class TestResult:
    def __init__(self, testname:str) -> None:
        self.testname:str = testname
        self.num_correct:int = 0
        self.num_wrong:int = 0
        self.badcases: list[tuple[Question,str]] = [] # list[(badcase1), badcase(2)]

    def update(self, iscorrect:bool, badcase:tuple[Question,str]=None) -> None:
        if iscorrect:
            self.num_correct += 1
        else:
            self.num_wrong += 1
            if badcase is None:
                logger.warning("badcase is not been restored.")
            else:
                assert isinstance(badcase, tuple) and isinstance(badcase[0], Question) and isinstance(badcase[1], str)
                self.badcases.append(badcase)


class Evaluator:
    def __init__(self) -> None:
        self._client = None
        self._eval_dataset = EvalDataset()
        self.results:dict[str: TestResult] = dict()
        self.current_testname:str = None
         
    def eval(self, testname:str, file_path:str|list[str]) -> None:
        async def warp_eval():
            logger.info(f"Stating task: {testname}")

            # create new TestResult
            self.current_testname = testname
            if testname not in self.results.keys():
                self.results[testname] = TestResult(testname)

            # load eval ds
            self._eval_dataset.load_from_file(file_path, clear_last=True)

            # create tasks
            pending = [
                asyncio.create_task(self._client.answer_question(q)) 
                for q in self._eval_dataset.question_list
                ]

            # send requests asynchonorize
            with tqdm(total=len(pending), desc=self.current_testname, unit="task") as bar:
                try:
                    while pending: # need a status bar
                        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                        # TODO: exceptiong check for res in done
                        # TODO: warning and delay for timeout

                        self.analyze_results([res.result() for res in done])
                        bar.update(len(done))
                    
                except KeyboardInterrupt:
                    # release resources
                    logger.error("User cancel all tasks!")
                    for t in pending:
                        t.cancel()

        # start event loop
        asyncio.run(warp_eval())


    def show_eval_results(self):
        print(f"############## Results ##############")
        for test in self.results.values():
            print(f"\nTest Name:{test.testname}")
            print(f"Correct:{test.num_correct}\tWrong:{test.num_wrong}\tACC:{test.num_correct/(test.num_wrong+test.num_correct)*100}%\n")
        print(f"#####################################")


    def save_result_as_csv(self, ouput_dir:str, file_name:str=None) -> None:
        """
        this will create csv in output_dir.
        if require_badcase is true a json file with same name will be created too.
        """
        raise NotImplementedError
        def get_formatted_time():
            now = datetime.now()
            formatted_time = now.strftime("%m%d%H%M")
            return formatted_time
        if file_name is None:
            file_name = f"eval_result_{get_formatted_time()}.csv"

    def load_from_file(self, file_path:str|list[str], clear_last=True) -> None:
        logger.info(f"Loading file to EvalDataset: {file_path}")
        self._eval_dataset.load_from_file(file_path, clear_last)

    def connect(self, base_url, model_name, api_key) -> None:
        """need with '/v1'"""
        self._client = AsyncClient(base_url, model_name, api_key)

    def analyze_results(self, results:tuple[Question, str]|Exception) -> None:
        for res in results:
            # res is a tuple: (Question, str)
            # TODO: ckeck exceptions
            q, r = res
            if q.check(r) == True:
                self.results[self.current_testname].update(True)
            else:
                self.results[self.current_testname].update(False, badcase=(q,r))


# if __name__ == "__main__":
#     """
#     使用方法:e.connect(base_url, 模型名称, apikey)
#             e.eval(测试名, 测试题位置)
#             e.show_eval_results()
#     """
#     e = Evaluator()
#     e.connect("http://127.0.0.1:9880/v1", "Qwen/Qwen1.5-0.5B-Chat-GGUF", "lm-studio")
#     e.eval(testname="MCQs_1", file_path="./testds0.json")
#     # e.eval(testname="MCQs_mix", file_path=["./testds1.json", "./testds2.json"])
#     e.show_eval_result()

if __name__ == "__main__":
    """
    目前只支持选择题.
    使用方法:e.connect(base_url, 模型名称, apikey)
            e.eval(测试名, 测试题位置)
            e.show_eval_results()
    """
    e = Evaluator()
    e.connect("http://127.0.0.1:9880/v1", "/data/zonepg/code/LLaMA-Factory/saves/Yi-1.5-9B-Chat/full/sft/QAs+ds+dialogue", "lm-studio")
    e.eval(testname="职业道德多选三级", file_path=["/data/lixubin/LLMToolBox/eval/questions/moral/multiple/third/11-18 职业道德多选.json", "/data/lixubin/LLMToolBox/eval/questions/moral/multiple/third/17-18 职业道德多选.json"])
    # e.eval(testname="职业道德多选二级", file_path=["/data/lixubin/LLMToolBox/eval/questions/moral/multiple/third/14 职业道德多选.json", "/data/lixubin/LLMToolBox/eval/questions/moral/multiple/third/15 职业道德多选.json", "/data/lixubin/LLMToolBox/eval/questions/moral/multiple/third/16 职业道德多选.json"])
    e.show_eval_results()