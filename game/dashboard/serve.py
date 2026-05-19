#!/usr/bin/env python3
"""
Minimal dev server for the Spindle MAPPO Training Dashboard.

Serves the dashboard static files AND the training_output directory
with appropriate CORS headers for local dev.

Usage:
    python3 serve.py [port]
    # Default port: 8384

Open http://localhost:8384/ in your browser.
"""

import http.server
import os
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8384

# Serve from the game/ directory so both dashboard/ and training_output/ are accessible
SERVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with CORS headers and custom MIME types."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=SERVE_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()

    def translate_path(self, path):
        """Route / to /dashboard/index.html, serve training_output as-is,
        and route all other requests under dashboard/."""
        # Strip query string for routing decisions
        clean = path.split('?')[0]
        if clean == '/' or clean == '':
            path = '/dashboard/index.html'
        elif clean.startswith('/dashboard/'):
            pass  # already under dashboard/
        elif clean.startswith('/training_output'):
            pass  # serve training data as-is
        else:
            # Everything else (css, js, etc.) lives under dashboard/
            path = '/dashboard' + path
        return super().translate_path(path)


if __name__ == '__main__':
    os.chdir(SERVE_DIR)
    with http.server.HTTPServer(('', PORT), CORSHandler) as httpd:
        print(f"\n  Spindle MAPPO Dashboard")
        print(f"  Serving from: {SERVE_DIR}")
        print(f"  Dashboard:    http://localhost:{PORT}/")
        print(f"  Metrics:      http://localhost:{PORT}/training_output/metrics.jsonl")
        print(f"  Matches:      http://localhost:{PORT}/training_output/matches/")
        print(f"\n  Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Shutting down.")
