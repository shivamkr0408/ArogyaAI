import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.safety.red_flags import check_red_flags


def run_test(name, symptom_list, raw_text, expect_match=None):
    result = check_red_flags(symptom_list, raw_text)
    if expect_match:
        status = "PASS" if result and result["matched_pattern"] == expect_match else "FAIL"
        detail = f"matched={result['matched_pattern'] if result else None} (expected {expect_match})"
    else:
        status = "PASS" if not result else "FAIL"
        detail = f"matched={result['matched_pattern'] if result else None} (expected no match)"
    print(f"[{status}] {name}: {detail}")


print("=" * 60)
print("TRUE POSITIVES - emergency phrasings (expect matches)")
print("=" * 60)

run_test(
    "Cardiac: chest pain + left arm pain",
    ["chest_pain", "left_arm_pain"],
    "I have chest pain and my left arm is hurting",
    expect_match="cardiac",
)

run_test(
    "Stroke: 'worst headache of my life'",
    [],
    "I have the worst headache of my life and can't seem to think clearly",
    expect_match="stroke",
)

run_test(
    "Respiratory: can't breathe, blue lips",
    ["difficulty_breathing", "blue_lips"],
    "I can barely breathe and my lips are turning blue",
    expect_match="respiratory_distress",
)

run_test(
    "Bleeding: vomiting blood",
    [],
    "I've been vomiting blood since this morning",
    expect_match="severe_bleeding",
)

run_test(
    "Anaphylaxis: throat swelling + breathing difficulty",
    ["breathing_difficulty", "throat_swelling"],
    "My throat is closing up and I can't breathe after eating peanuts",
    expect_match="anaphylaxis",
)

run_test(
    "Mental health: suicidal thoughts",
    [],
    "I don't want to live anymore",
    expect_match="mental_health_crisis",
)

print()
print("=" * 60)
print("FALSE POSITIVE CHECKS - similar but non-emergency phrasings")
print("=" * 60)

run_test(
    "Chest tightness from anxiety (mild)",
    ["chest_pain"],
    "I get a bit of chest tightness when I'm anxious but it passes quickly",
)

run_test(
    "Headache - regular migraine",
    ["headache"],
    "I have a headache today, maybe a migraine, it's not the worst I've had",
)

run_test(
    "Shortness of breath after climbing stairs",
    ["breathlessness"],
    "I get a little out of breath when climbing stairs but can still talk normally",
)

run_test(
    "Throat scratchy (cold)",
    ["throat_irritation"],
    "My throat is a bit scratchy from a cold but I can breathe fine",
)

run_test(
    "Regular fever and cough",
    ["high_fever", "cough"],
    "I have a fever and a cough for two days now",
)

print()
print("=" * 60)
print("Summary")
print("=" * 60)