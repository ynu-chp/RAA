
import os

def run_python_program(program_name):
    current_directory = os.getcwd()  # 获取当前目录
    print(current_directory)
    program_path = os.path.join(current_directory, program_name)  # 拼接程序路径
    print(program_path)
    # if not os.path.exists(program_path):
    #     print(f"Error: {program_name} not found in the current directory.")
    #     return
    #
    # if not program_name.endswith('.py'):
    #     print("Error: The specified file is not a Python program.")
    #     return

    os.system(f"/root/miniconda3/bin/python3.8 {program_path}")

if __name__ == "__main__":
    program_name =["2x5.py","4x5.py"]# input("请输入要运行的 Python 程序文件名（包括扩展名.py）：")
    for i in range(len(program_name)):
        run_python_program(program_name[i])



# if __name__ == "__main__":
#
#     script_path = ["2x5.py","4x5.py"]
#     for i in range(len(script_path)):
#         run_python_script(script_path[i])
