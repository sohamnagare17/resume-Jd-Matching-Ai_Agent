
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from strands import Agent
from strands.models.openai import OpenAIModel

from scoring import calculate_overall_score
from validator import validate_match_data
from file_manager import find_jd, find_resumes

load_dotenv()


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================



def extract_pdf_text(path):
    reader = PdfReader(path)

    pages = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text.strip())

    return "\n\n".join(pages)


# ============================================================
# JSON VALIDATION
# ============================================================

def validate_data(data):
    """
    Basic validation for structured extraction output.
    """

    if not isinstance(data, dict):
        raise ValueError("Extracted data is not a JSON object.")

    return data


# ============================================================
# ROBUST JSON RESPONSE PARSER
# ============================================================

def parse_json_response(response_text):
    """
    Parse JSON returned by the LLM.

    Handles:
    - pure JSON
    - markdown JSON code fences
    - explanatory text before/after the JSON object
    """

    response_text = str(response_text).strip()

    # Remove markdown fences if present.
    if response_text.startswith("```json"):
        response_text = response_text[len("```json"):].strip()
    elif response_text.startswith("```"):
        response_text = response_text[len("```"):].strip()

    if response_text.endswith("```"):
        response_text = response_text[:-3].strip()

    # First try the complete response.
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # LLMs sometimes add notes/explanations before the JSON.
    # Extract the outermost JSON object.
    start = response_text.find("{")
    end = response_text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        raise ValueError(
            "LLM response did not contain a JSON object.\n"
            f"Raw response:\n{response_text}"
        )

    json_text = response_text[start:end + 1]

    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM response contained JSON, but it was invalid: {e}\n"
            f"Extracted JSON:\n{json_text}\n\n"
            f"Raw response:\n{response_text}"
        )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

# Values can be overridden in a .env file without editing this source.
MODEL_ID = os.getenv("LLM_MODEL_ID")
MODEL_BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_API_KEY = os.getenv("LLM_API_KEY")
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS"))
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT")

model = OpenAIModel(
    model_id=MODEL_ID,
    client_args={
        "base_url": MODEL_BASE_URL,
        "api_key": MODEL_API_KEY,
    },
    params={
        "temperature": 0,
        "max_tokens": MAX_OUTPUT_TOKENS,
        # Qwen reasoning models may otherwise spend the output budget on
        # hidden reasoning. LM Studio supports "none" for compatible models.
        "reasoning_effort": REASONING_EFFORT,
    },
)


def invoke_model(prompt: str):
    """Run one independent task with fresh conversation history.

    Creating a new Agent for each independent JD/resume operation prevents
    previous resumes and reasoning messages from accumulating in later calls.
    """
    agent = Agent(model=model, callback_handler=None)
    return agent(prompt)


# ============================================================
# STRUCTURED EXTRACTION
# ============================================================

def extract_resume_data(resume_text):
    """
    Extract structured information from a Resume only.
    """

    extraction_prompt = """
You are a Resume information extraction agent.

Extract structured information ONLY from the provided Resume.

IMPORTANT RULES:

1. Extract only information explicitly present in the resume.
2. Do not invent information.
3. Do not assume experience duration.
4. Keep technical skills as individual skills.
5. Keep professional employment and internships under experience.
6. Keep domain-specific experience under domain_experience.
7. Keep education under education.
8. Keep projects under projects.
9. Do not treat projects as professional employment.
10. Return ONLY valid JSON.
11. Do not include markdown or explanations outside the JSON.
12. Keep each list item concise; do not copy the full resume into the output.
13. Keep each experience/project entry to the facts needed for matching.

Use exactly this JSON structure:

{
  "skills": [],
  "experience": [],
  "domain_experience": [],
  "education": [],
  "projects": []
}

RESUME:

"""

    extraction_prompt += resume_text

    print(f"      Resume text: {len(resume_text):,} characters")
    print("      Stage: resume extraction")
    response = invoke_model(extraction_prompt)

    response_text = str(response).strip()

    try:
        data = parse_json_response(response_text)
    except ValueError as e:
        raise ValueError(
            f"Resume extraction failed: {e}"
        )

    data = validate_data(data)

    required_fields = {
        "skills",
        "experience",
        "domain_experience",
        "education",
        "projects"
    }

    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise ValueError(
            "Resume extraction is missing fields: "
            + ", ".join(sorted(missing_fields))
        )

    return data


# ============================================================
# MATCHING
# ============================================================

def match_resume(jd_data, resume_data):
    """
    Compare JD requirements with Resume information.
    """

    # IMPORTANT:
    # This is a normal string, NOT an f-string.
    #
    # Therefore we do NOT have the JSON brace problem here.
    matching_prompt = """
You are a Resume-JD Matching Agent.

Your task is to compare the structured Job Description (JD) data with
the structured Resume data and determine how well the candidate's
information satisfies each JD requirement.

Use semantic comparison based only on the provided JD and Resume.

Do NOT invent information.
Do NOT assume experience duration.
Do NOT assume related technologies are equivalent.
Do NOT make hiring, rejection, or suitability decisions.

Return ONLY valid JSON.
Do not return markdown.
Do not return explanations outside the JSON.

--------------------------------------------------
OUTPUT SCHEMA
--------------------------------------------------

{
  "skills": {
    "matched": [],
    "partially_matched": [],
    "missing": [],
    "not_verified": []
  },
  "experience": {
    "matched": [],
    "partially_matched": [],
    "missing": [],
    "not_verified": []
  },
  "domain": {
    "matched": [],
    "partially_matched": [],
    "missing": [],
    "not_verified": []
  },
  "education": {
    "matched": [],
    "partially_matched": [],
    "missing": [],
    "not_verified": []
  },
  "evidence": [
    {
      "requirement": "",
      "category": "",
      "status": "",
      "reason": ""
    }
  ],
  "gaps": []
}

--------------------------------------------------
STATUS DEFINITIONS
--------------------------------------------------

MATCHED:

Use MATCHED when the resume clearly provides evidence that satisfies
the complete JD requirement.

Example:

JD:
"Java"

Resume:
"Java"

Result:
MATCHED


PARTIALLY_MATCHED:

Use PARTIALLY_MATCHED when the resume provides meaningful evidence
that satisfies only part of the requirement.

Example:

JD:
"Java and Selenium experience"

Resume:
"Java"

Result:
PARTIALLY_MATCHED


NOT_VERIFIED:

Use NOT_VERIFIED when the resume provides evidence that the underlying
skill or experience exists, but the exact JD requirement cannot be
verified.

This is especially important for duration requirements.

Example:

JD:
"2 years testing experience"

Resume:
"Software testing experience"

No duration is specified.

Result:
NOT_VERIFIED


MISSING:

Use MISSING when there is no meaningful evidence in the resume that
the requirement exists.

Example:

JD:
"Mainframe"

Resume:
No Mainframe-related information.

Result:
MISSING


--------------------------------------------------
NOT_VERIFIED VS MISSING
--------------------------------------------------

Use this distinction carefully.

If the resume provides evidence that the underlying skill or experience
exists but the exact requirement cannot be confirmed:

→ NOT_VERIFIED

If the resume provides no meaningful evidence:

→ MISSING

Example:

JD:
"5+ years Quality Assurance experience"

Resume:
"QA Tester" but no duration.

→ NOT_VERIFIED


JD:
"5+ years Quality Assurance experience"

Resume:
No QA/testing experience anywhere.

→ MISSING


--------------------------------------------------
EXPERIENCE DURATION
--------------------------------------------------

Skill presence and experience duration are separate concepts.

Never assume duration from simply seeing a skill.

Example:

JD:
"3 years Selenium experience"

Resume:
"Selenium WebDriver"

Result:
NOT_VERIFIED

Do NOT mark this MATCHED merely because Selenium is listed.

Do NOT mark it MISSING if Selenium experience is clearly indicated.


--------------------------------------------------
PROJECTS, INTERNSHIPS AND EMPLOYMENT
--------------------------------------------------

Distinguish between:

- Professional employment
- Internship
- Projects
- Skills
- Education

Do not automatically treat a project as professional employment.

Do not automatically treat an internship as equivalent to the required
years of professional experience.

Projects can provide evidence that a technology or testing activity
was used.

However, projects alone must not be used to invent professional
experience duration.


--------------------------------------------------
SKILL MATCHING
--------------------------------------------------

Match skills semantically when the resume clearly demonstrates the
same skill.

Related technologies are NOT automatically equivalent.

Examples:

Java → Java = MATCHED

Selenium WebDriver → Selenium = MATCHED

JavaScript → Java = NOT MATCHED

MySQL → Oracle = NOT automatically MATCHED

For multi-skill requirements, use PARTIALLY_MATCHED when only part of
the requirement is supported.


--------------------------------------------------
DOMAIN MATCHING
--------------------------------------------------

Evaluate domain experience separately from technical skills.

If the JD requires:

- Credit Card
- Commercial Banking
- Retail Banking
- Finance

and the resume does not mention those domains, classify them as MISSING.

Do not infer finance or banking experience merely because the candidate
has databases, APIs, payments, or generic business applications.

Use only explicit domain evidence from the resume.


--------------------------------------------------
EDUCATION
--------------------------------------------------

Evaluate education only when the JD contains an actual education
requirement.

If the JD contains no education requirement:

Keep all education lists empty:

"education": {
  "matched": [],
  "partially_matched": [],
  "missing": [],
  "not_verified": []
}

Do NOT create artificial requirements such as:

"Education requirements not specified"

Do NOT mark unspecified education requirements as MISSING.


--------------------------------------------------
REQUIREMENT STATUS CONSISTENCY
--------------------------------------------------

THIS RULE IS MANDATORY.

Every actual JD requirement MUST appear in exactly ONE status list.

A requirement MUST NEVER appear in multiple status lists.

INVALID:

"not_verified": [
  "2 years testing experience"
],
"missing": [
  "2 years testing experience"
]

The same requirement cannot appear twice.

Instead, select the single most appropriate status.

Example:

JD:
"2 years testing experience"

Resume:
"testing experience" with no duration.

Correct:

"not_verified": [
  "2 years testing experience"
]

Incorrect:

"not_verified": [
  "2 years testing experience"
],
"missing": [
  "2 years testing experience"
]


Before returning the JSON, internally check every category and ensure
that no requirement appears in multiple status lists.


--------------------------------------------------
EVIDENCE
--------------------------------------------------

For each evaluated JD requirement, provide an evidence object:

{
  "requirement": "",
  "category": "",
  "status": "",
  "reason": ""
}

category must be one of:

- skills
- experience
- domain
- education

status must exactly match the status assigned to that requirement.

Reasons must use only information from the resume.
Keep each reason to one concise sentence (ideally 20 words or fewer).
Do not quote large resume passages.

For MATCHED:
Explain why the requirement is satisfied.

For PARTIALLY_MATCHED:
Explain which part is supported and which part is not supported.

For NOT_VERIFIED:
Explain what evidence exists and what information is missing.

For MISSING:
Explain that no meaningful evidence was found.


--------------------------------------------------
GAPS
--------------------------------------------------

The gaps list should contain actual unmet or insufficiently verified
JD requirements.

Include:

- MISSING
- PARTIALLY_MATCHED
- NOT_VERIFIED

Do not include MATCHED requirements.

Do not create new requirements in the gaps list.

Use the same requirement wording used in the status lists whenever
possible.


--------------------------------------------------
NO INVENTION
--------------------------------------------------

Use only information contained in the JD and Resume.

Do not infer:

- years of experience
- job responsibilities
- domain experience
- technologies not mentioned
- certifications not mentioned
- education not mentioned
- professional experience from projects
- professional experience from skills alone

If information cannot be verified, use NOT_VERIFIED where appropriate.


--------------------------------------------------
FINAL SELF-CHECK
--------------------------------------------------

Before returning the response, verify:

1. Response is valid JSON.

2. Top-level sections are exactly:

   skills
   experience
   domain
   education
   evidence
   gaps

3. Every category contains:

   matched
   partially_matched
   missing
   not_verified

4. Every actual JD requirement has exactly ONE status.

5. No requirement appears in multiple status lists.

6. MATCHED requirements are not included in gaps.

7. MISSING, PARTIALLY_MATCHED, and NOT_VERIFIED requirements may be
   included in gaps.

8. Evidence status matches the corresponding classification.

9. Keep evidence reasons concise; do not repeat the resume.

10. No artificial requirements are created when a JD category has no
   requirements.

11. No experience duration is assumed.

12. No information is invented.

13. Return ONLY JSON.


--------------------------------------------------
JOB DESCRIPTION
--------------------------------------------------

{jd_data}


--------------------------------------------------
RESUME
--------------------------------------------------

{resume_data}
"""

    # Insert the actual data without using an f-string.
    matching_prompt = matching_prompt.replace(
        "{jd_data}",
        json.dumps(jd_data, ensure_ascii=False, indent=2)
    )

    matching_prompt = matching_prompt.replace(
        "{resume_data}",
        json.dumps(resume_data, ensure_ascii=False, indent=2)
    )

    print("      Stage: JD matching")
    response = invoke_model(matching_prompt)

    response_text = str(response).strip()

    try:
        match_data = parse_json_response(response_text)
    except ValueError as e:
        raise ValueError(
            f"Matching agent produced invalid JSON: {e}"
        )

    # Run our deterministic validator.
    validation_errors = validate_match_data(match_data)

    if validation_errors:
        error_message = "\n".join(
            f"- {error}" for error in validation_errors
        )

        raise ValueError(
            f"Matching agent produced invalid output.\n"
            f"{error_message}"
        )

    return match_data


# ============================================================
# REPORT
# ============================================================

def create_report(
    resume_path,
    jd_path,
    jd_data,
    resume_data,
    match_data,
    score
):
    """
    Create final candidate report.
    """

    return {
        "candidate": os.path.basename(resume_path),
        "job_description": os.path.basename(jd_path),

        "jd_data": jd_data,

        "resume_data": resume_data,

        "matching_result": match_data,

        "overall_match_percentage": score
    }


# ============================================================
# PROCESS ONE RESUME
# ============================================================

def process_resume(resume_path, jd_path, jd_data):
    """
    Process one resume using the already-extracted JD data.
    """

    resume_text = extract_pdf_text(resume_path)

    if not resume_text.strip():
        raise ValueError(
            f"No text could be extracted from resume: {resume_path}"
        )

    # Extract only resume information.
    resume_data = extract_resume_data(resume_text)

    # Match this resume against the already extracted JD.
    match_data = match_resume(
        jd_data,
        resume_data
    )

    # Scoring is deterministic Python logic, not an LLM decision.
    match_percentage = calculate_overall_score(match_data)

    return create_report(
        resume_path,
        jd_path,
        jd_data,
        resume_data,
        match_data,
        match_percentage
    )

# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print("     Resume-JD Matching Agent")
    print("======================================")
    print()

    # --------------------------------------------------------
    # Find JD
    # --------------------------------------------------------

    jd_path = find_jd()

    # --------------------------------------------------------
    # Find resumes
    # --------------------------------------------------------

    resume_files = find_resumes()


    # --------------------------------------------------------
    # Extract JD once
    # --------------------------------------------------------

    print("Extracting JD information...")

    jd_text = extract_pdf_text(jd_path)

    if not jd_text.strip():
        raise ValueError("JD PDF contains no extractable text.")

    jd_only_prompt = """
You are a Job Description information extraction agent.

Extract ONLY the structured information from the provided Job Description.

Do not extract resume information.

Do not invent information.

Return ONLY valid JSON.

Use exactly this structure:

{
  "job_title": "",
  "jd": {
    "required_skills": [],
    "preferred_skills": [],
    "experience_requirements": [],
    "domain_requirements": [],
    "education_requirements": []
  }
}

Rules:

1. Extract every explicit required technical skill separately.
2. Extract preferred skills separately.
3. Extract experience requirements separately.
4. Keep experience duration requirements intact.
5. Extract domain requirements separately.
6. Extract education requirements separately.
7. Do not infer missing information.
8. If a category has no requirements, return an empty list.

JOB DESCRIPTION:

"""

    jd_prompt = jd_only_prompt + jd_text

    print(f"JD text: {len(jd_text):,} characters; prompt: {len(jd_prompt):,} characters")
    print("Stage: JD extraction")
    jd_response = invoke_model(jd_prompt)

    jd_response_text = str(jd_response).strip()

    try:
        jd_data = parse_json_response(jd_response_text)
    except ValueError as e:
        raise ValueError(
            f"JD extraction failed: {e}"
        )

    jd_data = validate_data(jd_data)

    required_jd_fields = {
        "job_title",
        "jd"
    }

    missing_jd_fields = required_jd_fields - jd_data.keys()

    if missing_jd_fields:
        raise ValueError(
            "JD extraction is missing fields: "
            + ", ".join(sorted(missing_jd_fields))
        )

    print("JD extraction completed.")
    print()

    # --------------------------------------------------------
    # Process resumes
    # --------------------------------------------------------

    results = []
    failed_candidates = []
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    total = len(resume_files)

    for index, resume_path in enumerate(resume_files, start=1):

        filename = os.path.basename(resume_path)

        print(
            f"[{index}/{total}] {filename}...",
            end=" "
        )

        try:

            report = process_resume(
                resume_path=resume_path,
                jd_path=jd_path,
                jd_data=jd_data
            )

            score = report["overall_match_percentage"]

            results.append(report)

            # Save each successful report immediately. A later failure will not
            # erase completed work from this run.
            candidate_name = Path(report["candidate"]).stem
            output_path = os.path.join(output_dir, f"{candidate_name}_report.json")
            with open(output_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=2, ensure_ascii=False)

            print(f"✓ {score:.2f}%")

        except Exception as e:

            failed_candidates.append({"candidate": filename, "error": str(e)})
            print("✗ FAILED")
            print(f"      {str(e)}")

    # --------------------------------------------------------
    # No successful results
    # --------------------------------------------------------

    if not results:
        print("No resumes were processed successfully.")
        # Still write a summary so failures can be inspected after the run.
        summary = {
            "job_description": os.path.basename(jd_path),
            "total_resumes": total,
            "successfully_processed": 0,
            "failed": total,
            "failures": failed_candidates,
            "ranking": [],
        }
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)
        return

    # --------------------------------------------------------
    # Sort candidates
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["overall_match_percentage"],
        reverse=True
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Save individual reports
    # --------------------------------------------------------

    for report in results:

        candidate_name = os.path.splitext(
            report["candidate"]
        )[0]

        output_path = os.path.join(
            output_dir,
            f"{candidate_name}_report.json"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                report,
                file,
                indent=4,
                ensure_ascii=False
            )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = {
        "job_description": os.path.basename(jd_path),

        "total_resumes": total,

        "successfully_processed": len(results),

        "failed": total - len(results),

        "failures": failed_candidates,

        "ranking": [
            {
                "rank": index,
                "candidate": report["candidate"],
                "overall_match_percentage":
                    report["overall_match_percentage"]
            }

            for index, report in enumerate(
                results,
                start=1
            )
        ]
    }

    summary_path = os.path.join(
        output_dir,
        "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final CLI output
    # --------------------------------------------------------

    print()
    print("======================================")
    print("             Ranking")
    print("======================================")

    for index, report in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{report['candidate']} "
            f"→ "
            f"{report['overall_match_percentage']:.2f}%"
        )

    print()
    print(
        f"Completed: {len(results)}/{total}"
    )

    print(
        f"Reports saved in: {output_dir}/"
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
