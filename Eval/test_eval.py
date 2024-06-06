from src import *
import unittest


class TestEvaluator(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.e = Evaluator()

    def test_load_file_1(self):
        self.e.load_from_file("./testds1.json")
        self.assertTrue(self.e._eval_dataset.question_list[4].check("ABCD"))
        self.assertEqual(self.e._eval_dataset.question_list[6].question, "9．除了智慧性外，“诚信”职业道德规范的特征还包括（）。")
    
    def test_load_file_2(self):
        self.e.load_from_file(["./testds1.json","./testds2.json"])
        self.assertEqual(self.e._eval_dataset.size, 45+38)
        self.assertEqual(self.e._eval_dataset.question_list[-1].options["D"], "会仔细地听，但会一笑了之")
        self.assertEqual(str(self.e._eval_dataset.size), self.e._eval_dataset.question_list[-1].id)

    @unittest.skip
    def test_final(self):
        self.e.connect("http://127.0.0.1:9880/v1", "Qwen/Qwen1.5-0.5B-Chat-GGUF", "lm-studio")
        self.e.eval(testname="MCQs_1", file_path="./testds1.json")
        # self.e.eval(testname="MCQs_mix", file_path=["./testds1.json", "./testds2.json"])
        self.e.show_eval_results()
        # self.e.save_result_as_csv("./")


if __name__ == "__main__":
    unittest.main()