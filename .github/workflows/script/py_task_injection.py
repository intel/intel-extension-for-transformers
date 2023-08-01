import argparse
import re
from typing import Any

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--file_name", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--params", type=str, nargs='+')
args = parser.parse_args()


class InsertCode():
    def __init__(self, file_name: str, code_content: str, pattern: str) -> None:
        self.file_path = file_name
        self.code_content = code_content
        self.pattern = pattern
        

    def insert(self) -> None:
        original_code = self.get_source_code(self.file_path)
        if self.exists(original_code):
            print("[VAL INFO] code exists, reinsert")
            original_code = original_code.split("# insert code start")[0] + original_code.split(
                "# insert code end")[-1]

        replacement = r'\1\n{}\n'.format(self.code_content)
        new_code = re.sub(self.pattern, replacement, original_code)
        with open(self.file_path, 'w') as f:
            f.write(new_code)
        print("[VAL INFO] insert succeed")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.insert()

    def replace_params(self, params: str, params_flag: str) -> None:
        self.code_content = self.code_content.replace(params_flag, params)

    @staticmethod
    def get_source_code(file_path: str) -> str:
        with open(file_path, 'r') as f:
            original_code = f.read()
        return original_code

    @staticmethod
    def exists(target: str) -> bool:
        return "# insert code" in target


config = {
    "get_ITREX_cpu_memory_info": {
        "code_content": '''
        # insert code start
        import psutil
        memory_allocated = round(psutil.Process().memory_info().rss / 1024**3, 3)
        print("Iteration: " + str(i), "memory used total:", memory_allocated, "GB")
        # insert code end
        ''',
        "pattern":
        r'(tic = time\.time\(\))'
    }
}

if __name__ == "__main__":
    code_params = config.get(args.task, None)
    if code_params:
        code_inserter = InsertCode(args.file_name, code_params.get("code_content"), code_params.get("pattern"))
        if code_params.get("params") and code_params.get("params_flag"):
            for parameter, flag in zip(code_params.get("params"), code_params.get("params_flag")):
                code_inserter.replace_params(parameter, flag)
        code_inserter.insert()
    else:
        raise ValueError(f"invalid task: {args.task}")

