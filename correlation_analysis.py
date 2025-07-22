import pandas as pd
import json
import matplotlib.pyplot as plt

file_path = "SV_Students_SE3_2025_Python_Data_7_13.xlsx"
sheet_name = "36445"
df = pd.read_excel(file_path, sheet_name=sheet_name)

print(f"Columns in sheet {sheet_name}:", df.columns.tolist())

num_assignments_completed = df['assignment_code'].nunique()
total_attempts = len(df)
num_unique_submissions = df['submission_id'].nunique()
num_required_completed = df[df['is_required'] == 'required']['assignment_code'].nunique()
print(f"Assignments completed (unique assignment_code): {num_assignments_completed}")
print(f"Total attempts (rows): {total_attempts}")
print(f"Unique submissions (unique submission_id): {num_unique_submissions}")
print(f"Required assignments completed: {num_required_completed}")

def extract_correct_percent(val):
    try:
        d = json.loads(val) if isinstance(val, str) else {}
        return d.get("correct_percent", None)
    except Exception:
        return None

df["correct_percent"] = df["minihint_metrics"].apply(extract_correct_percent)

def tf_to_int(val):
    if val is True or val == "TRUE":
        return 1
    if val is False or val == "FALSE":
        return 0
    return None

for col in ["is_correct", "did_look_at_hint", "did_peek"]:
    df[col + "_num"] = df[col].apply(tf_to_int)


x1, y1 = df["attempt"], df["correct_percent"]
mask1 = x1.notna() & y1.notna()
corr1 = x1[mask1].corr(y1[mask1])
plt.figure()
plt.scatter(x1[mask1], y1[mask1], alpha=0.5)
plt.xlabel("Number of Attempts")
plt.ylabel("Correct Percent")
plt.title(f"Correlation: {corr1:.3f}")
plt.savefig("corr_attempt_vs_correct_percent_36445.png")
print(f"Correlation between attempt and correct_percent: {corr1:.3f}")

x2, y2 = df["attempt"], df["is_correct_num"]
mask2 = x2.notna() & y2.notna()
corr2 = x2[mask2].corr(y2[mask2])
plt.figure()
plt.scatter(x2[mask2], y2[mask2], alpha=0.5)
plt.xlabel("Number of Attempts")
plt.ylabel("Is Correct (1/0)")
plt.title(f"Correlation: {corr2:.3f}")
plt.savefig("corr_attempt_vs_is_correct_36445.png")
print(f"Correlation between attempt and is_correct: {corr2:.3f}")

x3, y3 = df["attempt"], df["did_look_at_hint_num"]
mask3 = x3.notna() & y3.notna()
corr3 = x3[mask3].corr(y3[mask3])
plt.figure()
plt.scatter(x3[mask3], y3[mask3], alpha=0.5)
plt.xlabel("Number of Attempts")
plt.ylabel("Did Look at Hint (1/0)")
plt.title(f"Correlation: {corr3:.3f}")
plt.savefig("corr_attempt_vs_did_look_at_hint_36445.png")
print(f"Correlation between attempt and did_look_at_hint: {corr3:.3f}")

x4, y4 = df["attempt"], df["did_peek_num"]
mask4 = x4.notna() & y4.notna()
corr4 = x4[mask4].corr(y4[mask4])
plt.figure()
plt.scatter(x4[mask4], y4[mask4], alpha=0.5)
plt.xlabel("Number of Attempts")
plt.ylabel("Did Peek (1/0)")
plt.title(f"Correlation: {corr4:.3f}")
plt.savefig("corr_attempt_vs_did_peek_36445.png")
print(f"Correlation between attempt and did_peek: {corr4:.3f}")

x5, y5 = df["attempt"], df["sketch_time"]
mask5 = x5.notna() & y5.notna()
corr5 = x5[mask5].corr(y5[mask5])
plt.figure()
plt.scatter(x5[mask5], y5[mask5], alpha=0.5)
plt.xlabel("Number of Attempts")
plt.ylabel("Sketch Time")
plt.title(f"Correlation: {corr5:.3f}")
plt.savefig("corr_attempt_vs_sketch_time_36445.png")
print(f"Correlation between attempt and sketch_time: {corr5:.3f}") 