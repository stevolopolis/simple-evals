"""
Copied directly from https://github.com/TIGER-AI-Lab/Program-of-Thoughts/blob/main/tool.py
"""
import func_timeout
import subprocess
def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x, {}, {})
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(60, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


def safe_execute_subprocess(code_string: str, keys=None, split_key="Answer: "):
    def execute(x):
        try:
            result = subprocess.run(
                ["python", "-c", x],    
                capture_output=True,
                text=True
            )
            if split_key is not None:
                return result.stdout.strip().split(split_key)[1]
            else:
                return result.stdout.strip()
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(60, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans