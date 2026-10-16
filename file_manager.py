from pathlib import Path


JD_DIR = Path("documents/jd")
RESUME_DIR = Path("documents/resumes")


def find_jd():
    jd_files = list(JD_DIR.glob("*.pdf"))

    if not jd_files:
        raise FileNotFoundError(
            "No JD PDF found in documents/jd/"
        )

    if len(jd_files) > 1:
        raise ValueError(
            "Multiple JD PDFs found. Please keep only one JD."
        )

    return jd_files[0]


def find_resumes():
    resume_files = list(RESUME_DIR.glob("*.pdf"))

    if not resume_files:
        raise FileNotFoundError(
            "No resume PDFs found in documents/resumes/"
        )

    return resume_files