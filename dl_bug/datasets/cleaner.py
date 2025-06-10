import json
import re

SRC  = "datasets/bug_report_raw.jsonl"
DEST = "datasets/bug_report.jsonl"

# TRACEBACK CLEANING
# If the code contains Traceback, remove it from the Code 
# and append it to Bug Description
traceback_re = re.compile(r"^\s*Traceback", re.I)        # match first “Traceback” line (case-insensitive)

def process(obj: dict) -> dict | None:
    """
    Strip traceback from obj['Code'], append it to obj['Bug Description'].
    Return the modified object, or None if Code ends up empty.
    """
    code_lines = obj.get("Code", "").splitlines()

    for i, line in enumerate(code_lines):
        if traceback_re.match(line):
            # split code vs traceback
            code_only      = "\n".join(code_lines[:i]).rstrip()
            traceback_part = "\n".join(code_lines[i:]).strip()

            # update fields
            obj["Code"] = code_only
            obj["Bug Description"] = (
                obj.get("Bug Description", "").rstrip() +
                (" | " if obj.get("Bug Description") else "") +
                traceback_part
            )
            break

    # drop record if Code is now empty / whitespace
    if not obj.get("Code", "").strip():
        return None
    return obj


with open(SRC, encoding="utf-8") as fin, \
     open(DEST, "w", encoding="utf-8") as fout:

    for raw_line in fin:
        try:
            record = json.loads(raw_line)
        except json.JSONDecodeError as err:
            print(f"Skipping malformed JSON: {err}")
            continue

        cleaned = process(record)
        if cleaned is not None:
            fout.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

print(f"Cleaning complete. Output written to {DEST}")