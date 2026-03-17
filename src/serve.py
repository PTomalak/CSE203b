import os
import http.server
import socketserver
import re

PORT = 8000

class CORS_Range_RequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Range')
        self.send_header('Access-Control-Expose-Headers', 'Content-Range, Accept-Ranges, Content-Length')
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if not os.path.isfile(path) or 'Range' not in self.headers:
            return super().do_GET()

        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(http.HTTPStatus.NOT_FOUND, "File not found")
            return

        size = os.path.getsize(path)
        match = re.search(r'bytes=(\d+)-(\d*)', self.headers['Range'])
        if match:
            first = int(match.group(1))
            last = int(match.group(2)) if match.group(2) else size - 1
            length = last - first + 1
            
            self.send_response(206)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()
            
            f.seek(first)
            self.wfile.write(f.read(length))
            f.close()
            return

with socketserver.TCPServer(("", PORT), CORS_Range_RequestHandler) as httpd:
    print(f"Serving at http://localhost:{PORT} with CORS and Partial-Range support")
    httpd.serve_forever()
