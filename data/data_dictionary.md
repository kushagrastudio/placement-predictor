# Data Dictionary

Source: Kaggle — Student Placement Prediction Dataset 2026
Verified 12 Sep: 100,000 rows, 26 columns, 0 missing values, 0 duplicates.

| Column | Type | Values / Range | Notes |
|---|---|---|---|
| student_id | int | 1–100000 | Unique identifier. Dropped before modeling. |
| age | int | 18–24 | |
| gender | categorical (nominal) | Male, Female | One-hot encoded |
| cgpa | float | ~5–10 | |
| branch | categorical (nominal) | CSE, IT, ECE, EEE, Mechanical, Civil | One-hot encoded |
| college_tier | categorical (ordinal) | Tier 1, Tier 2, Tier 3 | Ordinal encoded (Tier 1 = most prestigious) |
| internships_count | int | count | |
| projects_count | int | count | |
| certifications_count | int | count | |
| coding_skill_score | float | ~0–100 | |
| aptitude_score | float | ~0–100 | |
| communication_skill_score | float | ~0–100 | |
| logical_reasoning_score | float | ~0–100 | |
| hackathons_participated | int | count | |
| github_repos | int | count | |
| linkedin_connections | int | count | |
| mock_interview_score | float | ~0–100 | |
| attendance_percentage | float | ~0–100 | |
| backlogs | int | 0–6 | See docs/eda_findings.md |
| extracurricular_score | float | ~0–100 | |
| leadership_score | float | ~0–100 | |
| volunteer_experience | categorical (nominal) | Yes, No | One-hot encoded |
| sleep_hours | float | hours | |
| study_hours_per_day | float | hours | |
| placement_status | target 1 | Placed, Not Placed | 54.5% / 45.5% split |
| salary_package_lpa | target 2 | 0.0 or 7.11–20.44 | Excluded from classifier features |