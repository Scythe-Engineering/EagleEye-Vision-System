from flask import send_from_directory


def serve_index():
    """
    Serve the index.html file.

    Returns:
        Response: The index.html file.
    """
    return send_from_directory("./static", "index.html")


def serve_js():
    """
    Serve the JavaScript file.

    Returns:
        Response: The JavaScript file.
    """
    return send_from_directory("./static", "bundle.js")


def serve_css():
    """
    Serve the CSS file.

    Returns:
        Response: The CSS file.
    """
    return send_from_directory("./static", "main.css")
