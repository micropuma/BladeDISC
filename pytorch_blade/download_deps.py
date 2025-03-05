import os
import re
import requests
import hashlib
from urllib.parse import urlparse

# 配置参数
WORKSPACE_FILE = "WORKSPACE"  # 原始 WORKSPACE 文件路径
OUTPUT_WORKSPACE = "WORKSPACE.offline"  # 生成的离线 WORKSPACE 文件
DEPS_DIR = "offline_deps"  # 依赖保存目录
MIRROR_URL = None  # 可选：内网镜像源前缀，如 "http://internal-mirror/deps/"

os.makedirs(DEPS_DIR, exist_ok=True)

def sha256sum(file_path):
    """计算文件的 SHA256 哈希值"""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_file(url, dest_path):
    """下载文件并校验完整性"""
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"Download failed: {e}")
        return False

def process_workspace():
    """处理 WORKSPACE 文件"""
    with open(WORKSPACE_FILE, "r") as f:
        content = f.read()

    # 匹配 http_archive 和 blade_http_archive 规则
    pattern = r"(?P<rule>(http_archive|blade_http_archive)\s*$\s*name\s*=\s*\"(?P<name>[^\"]+)\".*?urls\s*=\s*$$(?P<urls>.*?)$$\s*.*?$)"
    matches = re.finditer(pattern, content, re.DOTALL)

    offline_content = content
    for match in matches:
        rule = match.group("rule")
        name = match.group("name")
        urls = [url.strip(" '\"") for url in match.group("urls").split(",") if url.strip()]

        # 下载文件
        downloaded = False
        local_path = None
        for url in urls:
            if MIRROR_URL and url.startswith("http"):
                # 使用内网镜像源
                filename = os.path.basename(urlparse(url).path)
                mirror_url = f"{MIRROR_URL}{filename}"
                urls = [mirror_url] + urls  # 优先使用镜像源

            filename = os.path.basename(urlparse(url).path)
            dest_path = os.path.join(DEPS_DIR, filename)

            if os.path.exists(dest_path):
                # 检查哈希是否匹配
                sha256_match = re.search(r'sha256\s*=\s*"([^"]+)"', rule)
                if sha256_match:
                    expected_sha = sha256_match.group(1)
                    actual_sha = sha256sum(dest_path)
                    if actual_sha != expected_sha:
                        print(f"SHA256 mismatch for {filename}, redownloading...")
                        os.remove(dest_path)
                    else:
                        print(f"Using cached {filename}")
                        downloaded = True
                        local_path = f"file://{os.path.abspath(dest_path)}"
                        break

                else:
                    print(f"Using cached {filename} (no SHA256 check)")
                    downloaded = True
                    local_path = f"file://{os.path.abspath(dest_path)}"
                    break

            if download_file(url, dest_path):
                downloaded = True
                local_path = f"file://{os.path.abspath(dest_path)}"
                break

        if not downloaded:
            print(f"⚠️ Failed to download {name}")
            continue

        # 替换 URLs
        new_rule = re.sub(
            r'urls\s*=\s*$$.*?$$',
            f'urls = ["{local_path}"]',
            rule,
            flags=re.DOTALL
        )
        offline_content = offline_content.replace(rule, new_rule)

    # 保存生成的 WORKSPACE 文件
    with open(OUTPUT_WORKSPACE, "w") as f:
        f.write(offline_content)
    print(f"✅ 离线版 WORKSPACE 已生成: {OUTPUT_WORKSPACE}")

if __name__ == "__main__":
    process_workspace()
