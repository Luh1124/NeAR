from huggingface_hub import hf_hub_download

repo_id = "luh0502/NeAR"
subfolder = "ref_images"

from huggingface_hub import HfApi

api = HfApi()
files_in_folder = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision="main")

download_dir = "./downloaded_ref_images"
import os
os.makedirs(download_dir, exist_ok=True)

for file_name in files_in_folder:
    if file_name.startswith(subfolder + "/"): 
        relative_path = os.path.relpath(file_name, subfolder) 
        local_path = os.path.join(download_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True) 
        hf_hub_download(repo_id=repo_id, repo_type="dataset", subfolder=subfolder, filename=relative_path, local_dir=download_dir)
        print(f"Downloaded: {file_name} to {local_path}")

print(f"所有 {subfolder} 文件夹下的文件已下载到 {download_dir}/ 目录。")
