import numpy as np
import matplotlib.pyplot as plt
import os

# ===== 参数 =====
N = 100000   # 读取最后 N 个点

# ===== 读取文件 =====
file_path = os.path.join(os.getcwd(), "curr.data")

if not os.path.exists(file_path):
    raise RuntimeError(f"[ERROR] File not found: {file_path}")

data = np.loadtxt(file_path)

if data.shape[1] < 5:
    raise RuntimeError("[ERROR] Need at least 5 columns")

x = data[:, 0]
y = data[:, 4]

# ===== 截取最后 N 点 =====
if N > len(x):
    print("[WARNING] N larger than data, using full data")
    N = len(x)

x = x[-N:]
y = y[-N:]

print(f"[INFO] Using last {N} points")

# ===== 画图 =====
plt.figure()
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Raw data")
plt.savefig("xy.png", dpi=300)
plt.close()

# ===== 平滑（关键！防止假峰）=====
window = max(5, N // 100)  # 自适应窗口
y_smooth = np.convolve(y, np.ones(window)/window, mode='same')

# ===== 找峰（更稳健）=====
try:
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(y_smooth, distance=len(y)//10)
    print(f"[INFO] Peaks found: {len(peaks)}")
except:
    print("[WARNING] scipy not available, fallback to simple method")
    peaks = []
    for i in range(1, len(y_smooth)-1):
        if y_smooth[i] > y_smooth[i-1] and y_smooth[i] > y_smooth[i+1]:
            peaks.append(i)

# ===== 选择周期 =====
if len(peaks) >= 2:
    start = peaks[-2] + 1   # 去掉第一个峰
    end = peaks[-1]

    if start < end:
        y_cycle = y[start:end]
        print(f"[INFO] Using cycle: {start} → {end}")
    else:
        print("[WARNING] Bad peak interval, fallback to full data")
        y_cycle = y
else:
    print("[WARNING] Not enough peaks, fallback to full data")
    y_cycle = y

# ===== 再保险 =====
if len(y_cycle) == 0:
    print("[WARNING] Empty cycle, fallback to full data")
    y_cycle = y

# ===== 求平均 =====
y_avg = np.mean(y_cycle)

print(f"[RESULT] Average y = {y_avg:.10e}")

# ===== 写文件（保证一定执行）=====
with open("y.txt", "w") as f:
    f.write(f"{y_avg:.10e}\n")

print("[INFO] y.txt written successfully")
