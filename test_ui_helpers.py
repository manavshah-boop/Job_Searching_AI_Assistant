import dashboard


def test_lines_to_list_strips_blank_lines_and_whitespace():
    assert dashboard._lines_to_list("  Python  \n\n AWS\n   \nLLM systems  ") == [
        "Python",
        "AWS",
        "LLM systems",
    ]


def test_build_jobs_table_frame_returns_expected_columns_and_values():
    frame = dashboard.build_jobs_table_frame(
        [
            {
                "id": "job-1",
                "title": "Backend Engineer",
                "company": "Acme",
                "location": "",
                "source_label": "Greenhouse",
                "status_label": "Applied",
                "score_state": "Scored",
                "fit_score": 88,
                "ats_score": 76,
                "one_liner": "Strong match for Python backend work.",
                "url": "https://example.com/job-1",
            }
        ]
    )

    assert list(frame.columns) == [
        "id",
        "Title",
        "Company",
        "Location",
        "Source",
        "Job status",
        "Score state",
        "Fit",
        "ATS",
        "Summary",
        "Posting",
    ]
    assert frame.iloc[0].to_dict() == {
        "id": "job-1",
        "Title": "Backend Engineer",
        "Company": "Acme",
        "Location": "Location not listed",
        "Source": "Greenhouse",
        "Job status": "Applied",
        "Score state": "Scored",
        "Fit": 88,
        "ATS": 76,
        "Summary": "Strong match for Python backend work.",
        "Posting": "https://example.com/job-1",
    }
    assert str(frame["Fit"].dtype) == "Int64"
    assert str(frame["ATS"].dtype) == "Int64"
