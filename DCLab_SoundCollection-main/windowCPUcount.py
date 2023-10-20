import psutil
import time

pid = 30104  # 대상 프로세스의 PID로 대체
process = psutil.Process(pid)

interval = 1  # 각 시간 단계의 간격 (초)
total_cpu_percent = 0  # CPU 사용량의 누적 값
num_samples = 0  # 샘플 수

while True:
    cpu_percent = process.cpu_percent(interval=interval)
    total_cpu_percent += cpu_percent
    num_samples += 1

    print(f'CPU 사용량: {cpu_percent}%')

    # 일정 시간이 지나면 평균을 계산하고 초기화
    if num_samples >= 10:  # 예를 들어, 10개의 샘플을 사용하여 평균 계산
        avg_cpu_percent = total_cpu_percent / num_samples
        print(f'평균 CPU 사용량 (최근 {num_samples}개 샘플): {avg_cpu_percent}%')

        # 누적 값을 초기화
        total_cpu_percent = 0
        num_samples = 0

    time.sleep(3)  # 샘플링 간격과 일정 시간 간격을 조절하실 수 있습니다
