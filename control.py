import os
from time import sleep
import json
from multiprocessing import Process
states_json_path = "/workspace/go_proj/src/Ai_WebServer/algorithm_utils/model_states.json"
model_name = "realesrgan"

def run_model_server():
    os.system("python run.py")

if __name__ == '__main__':
    running = False
    while True:
        with open(states_json_path,"r",encoding="utf-8")as f:
            states = json.load(f)
            if states[model_name] and not running:
                os.system("ps -ef|grep python|grep -v control|cut -c 9-16|xargs kill -9")
                p = Process(target=run_model_server,args=())
                p.start()
                running = True
            elif not states[model_name]:
                running = False
        sleep(1)