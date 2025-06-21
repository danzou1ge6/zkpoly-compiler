import re
import matplotlib.pyplot as plt
import seaborn as sns

# 读取文件
filename = 'bandwidth.txt'
with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 解析数据
bandwidth_dict = {}
pattern = re.compile(r'([\w\s]+)bandwidth:\s*([\d.]+)GB/s')

for line in lines:
    m = pattern.search(line)
    if m:
        bw_type = m.group(1).strip()  # 例如 'CPU to Disk'
        bw_value = float(m.group(2))  # 带宽数值
        bandwidth_dict.setdefault(bw_type, []).append(bw_value)

# 绘制PDF（概率密度分布）
plt.figure(figsize=(7, 5))
for i, (bw_type, values) in enumerate(bandwidth_dict.items()):
    if len(values) > 1:
        sns.kdeplot(values, fill=True, label=bw_type)
plt.legend()
plt.title('Bandwidth PDF for All Types')
plt.xlabel('Bandwidth (GB/s)')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.savefig("bandwidth_pdf.png", dpi=150)  # 保存图片
plt.close()

print("带宽分布PDF图片已保存为 bandwidth_pdf.png")