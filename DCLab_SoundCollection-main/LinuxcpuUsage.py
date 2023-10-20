import os
import time
import sys
import psutil
import time
import subprocess

if __name__ == "__main__":
    pid = int(sys.argv[1])
    py  = psutil.Process(pid)
    total = 0
    i = 0
    print("##############")
    print("PID: ",pid)
    print("##############")
    while True:
        temp = subprocess.run(["ps", "-p",str(pid),"-o","pcpu"], stdout=subprocess.PIPE, text=True)
        text = (temp.stdout).split("\n")
        cpu_usage = float(text[1])

        total += cpu_usage
        i += 1
        print("\n")
        print("cpu usage\t\t:", cpu_usage, "%")
        print("avg cpu usage\t\t:",total/i,"%")
        
        time.sleep(3)
        