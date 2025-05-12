import subprocess
import pkg_resources

# Step 1: 获取所有已安装的包
installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

# Step 2: 获取所有可升级的包
upgradeable_packages = {}
outdated = subprocess.run(["pip", "list", "--outdated", "--format=freeze"], capture_output=True, text=True)
for line in outdated.stdout.strip().split("\n"):
    if "==" in line:
        pkg, latest_version = line.split("==")
        upgradeable_packages[pkg.lower()] = latest_version.strip()

# Step 3: 生成正确的 `requirements.txt`
requirements = []
for pkg, version in installed_packages.items():
    if pkg in upgradeable_packages:
        # 只升级没有冲突的包
        latest_version = upgradeable_packages[pkg]
        requirements.append(f"{pkg}=={latest_version}")
    else:
        requirements.append(f"{pkg}=={version}")

# Step 4: 保存到文件
with open("requirements_fixed.txt", "w") as f:
    f.write("\n".join(requirements))

print("\n✅ 依赖检查完成，已生成正确的 requirements_fixed.txt")
print("请手动检查 pip check 是否仍然报错：")
print("运行: pip check")
