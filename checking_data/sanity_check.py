import json

# paths to files
conv_path = "../data_conversation/train_conversations.json"
qcm_path  = "../data_questions/train_formatted.json"

# Try loading a few records from the conversation file
with open(conv_path, "r") as f:
    conv_data = json.load(f)
print("Number of conversation examples:", len(conv_data))
print("Example #0 keys:", conv_data[0].keys())
print("Example #0 (instruction / input / output):")
print("  instruction:", conv_data[0]["instruction"])
print("  input      :", conv_data[0]["input"][:200], "…")
print("  output     :", conv_data[0]["output"][:200], "…")
print()

# Try loading a few records from the MCQ file
with open(qcm_path, "r") as f:
    qcm_data = json.load(f)
print("Number of MCQ examples:", len(qcm_data))
print("Example #0 keys:", qcm_data[0].keys())
print("Example #0 (instruction / input / output):")
print("  instruction:", qcm_data[0]["instruction"])
print("  input      :", qcm_data[0]["input"][:200], "…")
print("  output     :", qcm_data[0]["output"][:200], "…")
