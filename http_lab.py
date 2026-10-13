"""Loopback-only fault lab. Never targets an external service."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from contextlib import contextmanager

FAULTS = ('database', 'queue', 'dependency')
PROBES = ('db_probe', 'queue_probe', 'dependency_probe')
# Observable service symptoms: each probe is a real HTTP request to the local lab.
RESPONSES = {
    'database': {'db_probe': 503, 'queue_probe': 200, 'dependency_probe': 200},
    'queue': {'db_probe': 200, 'queue_probe': 429, 'dependency_probe': 200},
    'dependency': {'db_probe': 200, 'queue_probe': 200, 'dependency_probe': 504},
}

class LabServer(ThreadingHTTPServer):
    allow_reuse_address = True
    def __init__(self, address, fault):
        if fault not in FAULTS:
            raise ValueError('Unknown fault')
        self.fault = fault
        super().__init__(address, LabHandler)

class LabHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        probe = self.path.removeprefix('/')
        if probe not in PROBES:
            code, payload = 404, {'error': 'unknown probe'}
        else:
            code = RESPONSES[self.server.fault][probe]
            payload = {'probe': probe, 'healthy': code == 200}
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *_):
        pass

@contextmanager
def local_lab(fault):
    server = LabServer(('127.0.0.1', 0), fault)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown()
        worker.join(timeout=3)
        server.server_close()

def probe(base_url, probe_name):
    if probe_name not in PROBES:
        raise ValueError('Unknown probe')
    # Guard against accidental probing of any remote host.
    from urllib.parse import urlsplit
    url = urlsplit(base_url)
    if url.hostname != '127.0.0.1' or url.scheme != 'http' or url.username or url.password:
        raise ValueError('Only 127.0.0.1 HTTP lab is permitted')
    from urllib.error import HTTPError
    try:
        with urlopen(base_url + '/' + probe_name, timeout=2) as response:
            return response.status
    except HTTPError as exc:
        return exc.code
