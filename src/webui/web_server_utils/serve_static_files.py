from pathlib import Path

from flask import send_from_directory


STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


def serve_index():
    """
    Serve the index.html file.

    Returns:
        Response: The index.html file.
    """
    return send_from_directory(str(STATIC_DIR), "index.html")


def serve_js():
    """
    Serve the JavaScript file.

    Returns:
        Response: The JavaScript file.
    """
    return send_from_directory(str(STATIC_DIR), "bundle.js")


def serve_css():
    """
    Serve the CSS file.

    Returns:
        Response: The CSS file.
    """
    return send_from_directory(str(STATIC_DIR), "main.css")
