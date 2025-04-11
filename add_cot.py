import json

# Load generated_full.jsonl into a dictionary for quick lookup
with open("data/tofu/retain10.json", "r", encoding="utf-8") as f:
    generated_data = [json.loads(line) for line in f]
    generated_dict = {item["question"]: item["cot"] for item in generated_data}

# Load forget100.json and update its cot values
updated_forget_data = []
with open("data/tofu/real_authors_perturbed.json", "r", encoding="utf-8") as f:
    forget_data = [json.loads(line) for line in f]
    
    for item in forget_data:
        question = item["question"]
        if question in generated_dict:
            item["cot"] = generated_dict[question]  # Replace with corresponding generated_cot
        updated_forget_data.append(item)

# Save the updated forget100.json
with open("data/tofu/real_authors_perturbed.json", "w", encoding="utf-8") as f:
    for item in updated_forget_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("forget100.json has been updated and saved as forget100_updated.json.")