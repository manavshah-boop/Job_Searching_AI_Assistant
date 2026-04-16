"""
Compatibility Streamlit entrypoint.

`dashboard.py` is the real dashboard entrypoint. This file remains so environments
that auto-detect `app.py` still open the redesigned dashboard instead of the old
configuration editor.
"""

from dashboard import main as dashboard_main


def main():
    dashboard_main()


if __name__ == "__main__":
    main()
