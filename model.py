# from huggingface_hub import create_repo

# repo_url = create_repo(repo_id="LRM-unlearning-target-5epochs", private=False)
# print(f"Repo created at: {repo_url}")

from huggingface_hub import HfApi
import os

# Hugging Face repo 정보
repo_id = "sangyon/LRM-unlearning-forget10-retrain"  
folder_path = "results/steps/llama3-8b/forget10/RETRAIN/seed_1001/epoch10_1e-05_FixRef_maskTrue_1.0_1.0/1/unlearn_times_1/checkpoint-last"

api = HfApi()

# checkpoint last 폴더 내 파일 다 가져오기
files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# 파일 업로드
for file in files:
    print(f"Uploading {file} ...")
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=os.path.basename(file),  
        repo_id=repo_id,
        repo_type="model"
    )

print("Upload Complete!")