# 定义输出目录
import os
import json
data = json.loads(open("../config.json").read())
output_dir = "../storage/regions_output"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个区域，将每个区域保存为单独的 JSON 文件
for region in data["regions"]:
    # 生成区域文件名，使用区域名称命名文件
    file_name = f"{region['name']}.json"
    file_path = os.path.join(output_dir, file_name)

    # 将区域信息保存到 JSON 文件
    with open(file_path, "w") as f:
        json.dump(region, f, indent=4)

    print(f"区域 '{region['name']}' 的完整 JSON 数据已成功保存到文件 {file_path} 中。")