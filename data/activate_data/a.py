import glob

# 使用 glob 搜索当前目录下包含 'activate' 的 .txt 文件
files = glob.glob('*activate*.txt')

# 输出找到的文件名
print("Found files:")
smiles = []
for file in files:
    print(file)
    with open(file,"r") as f:
        for i in f:
            if i.strip():
                smiles.append(i.strip())

print(len(smiles))
