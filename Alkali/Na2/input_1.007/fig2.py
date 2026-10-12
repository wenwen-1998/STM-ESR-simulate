import numpy as np
import matplotlib.pyplot as plt

# ================== 参数区 ==================
DATA_PATH = "curr.data"
TAIL_N = 100000
smooth_window = 11
amp_frac = 0.98
# ===========================================

# ---------- 读取数据 ----------
data = np.loadtxt(DATA_PATH)

x = data[:, 0]
y = data[:, 4]      # ⚠️ 若量级不对，只需改这一列

x_tail = x[-TAIL_N:]
y_tail = y[-TAIL_N:]

y_max = y_tail.max()
y_min = y_tail.min()
mean_val = (y_max + y_min) / 2
amp = (y_max - y_min) / 2

# ---------- 平滑（可选） ----------
if smooth_window > 1:
    w = smooth_window + (smooth_window % 2 == 0)
    kernel = np.ones(w) / w
    y_use = np.convolve(y_tail, kernel, mode="same")
else:
    y_use = y_tail.copy()

# ---------- 找局部最小值 ----------
cand = np.where(
    (y_use[1:-1] < y_use[:-2]) &
    (y_use[1:-1] <= y_use[2:])
)[0] + 1

# 振幅阈值，去噪：只保留足够低的谷值
threshold = mean_val - amp_frac * amp
valleys = cand[y_use[cand] <= threshold]

# 谷值对应的 x / y（y 用原始值）
x_valleys = x_tail[valleys]
y_valleys = y_tail[valleys]

# ---------- 去掉第一个和最后一个最小值 ----------
if y_valleys.size >= 3:
    x_valleys_mid = x_valleys[1:-1]
    y_valleys_mid = y_valleys[1:-1]
else:
    x_valleys_mid = np.array([])
    y_valleys_mid = np.array([])

# ---------- 周期最小值平均 ----------
num_valleys_mid = y_valleys_mid.size
Iaver_valleys = float(np.mean(y_valleys_mid)) if num_valleys_mid > 0 else np.nan

# ---------- 输出结果 ----------
with open("y_min_stats.txt", "w") as f:
    f.write(f"tail_N = {TAIL_N}\n")
    f.write(f"total_valleys = {y_valleys.size}\n")
    f.write(f"used_valleys = {num_valleys_mid}  # removed first & last\n")
    f.write(f"Iaver_valleys = {Iaver_valleys}\n")

# ---------- 保存原始谷值 ----------
np.savetxt(
    "cycle_min_vals_mid.txt",
    y_valleys_mid,
    fmt="%.16e",
    header="cycle_min_values (first & last removed)",
    comments=""
)

# ---------- 画图 ----------
plt.figure(figsize=(6, 4), dpi=150)
plt.plot(x_tail, y_tail, lw=1.0, label="y (raw)")
plt.scatter(x_valleys, y_valleys, s=12, label="all valleys", alpha=0.5)
plt.scatter(x_valleys_mid, y_valleys_mid, s=18, label="used valleys", color="red")

plt.xlabel("x")
plt.ylabel("y")
plt.title(f"Iaver_valleys = {Iaver_valleys:.3e}")
plt.legend()
plt.tight_layout()
plt.savefig("last_current_min.png")
plt.show(block=False)
plt.pause(1)
plt.close()

