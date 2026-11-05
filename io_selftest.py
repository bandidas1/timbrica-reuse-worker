# Self-test of the worker's transfer rules against a local HTTP server — no GPU, runs in CI before the image ships.
# A red line here means a finished enhancement would be thrown away on the first dropped connection.
import http.server
import os
import tempfile
import threading
import time

import reuse_io

reuse_io._sleep_for = lambda attempt: 0.01   # the rules, not the waiting
STATE = {'put_fail': 0, 'put_mode': '', 'got': b'', 'get_fail': 0, 'puts': 0}
PAYLOAD = b'fLaC' + os.urandom(200_000)


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if STATE['get_fail'] > 0:
            STATE['get_fail'] -= 1
            self.send_response(503); self.end_headers(); return
        if self.path.startswith('/gone'):
            self.send_response(410); self.end_headers(); return
        self.send_response(200); self.send_header('Content-Length', str(len(PAYLOAD))); self.end_headers()
        self.wfile.write(PAYLOAD)

    def do_PUT(self):
        STATE['puts'] += 1
        n = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(n)
        if STATE['put_fail'] > 0:
            STATE['put_fail'] -= 1
            if STATE['put_mode'] == 'drop':
                self.connection.close(); return
            self.send_response(503); self.end_headers(); return
        if self.path.startswith('/big'):
            self.send_response(413); self.end_headers(); return
        STATE['got'] = body
        self.send_response(204); self.end_headers()


srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
base = 'http://127.0.0.1:%d' % srv.server_port
fails = 0


def ok(c, m):
    global fails
    print(('  ok  ' if c else '  FAIL ') + m)
    if not c:
        fails += 1


with tempfile.TemporaryDirectory() as td:
    src = os.path.join(td, 'src')
    STATE['get_fail'] = 2
    n = reuse_io.download(base + '/src', src)
    ok(n == len(PAYLOAD) and open(src, 'rb').read() == PAYLOAD, 'GET survives two 503s and returns the whole file')
    try:
        reuse_io.download(base + '/gone', src); ok(False, 'GET 410 must be final')
    except reuse_io.TransferError as e:
        ok(e.code == 'source_http_410', 'GET 410 is final (no retry): ' + e.code)

    out = os.path.join(td, 'out.flac'); open(out, 'wb').write(PAYLOAD)
    STATE.update(put_fail=3, put_mode='503', puts=0)
    st = reuse_io.upload(base + '/res', out)
    ok(st == 204 and STATE['got'] == PAYLOAD and STATE['puts'] == 4, 'PUT retries 503 ×3 and lands the same bytes (%d tries)' % STATE['puts'])
    STATE.update(put_fail=2, put_mode='drop', puts=0, got=b'')
    st = reuse_io.upload(base + '/res', out)
    ok(st == 204 and STATE['got'] == PAYLOAD, 'PUT survives two dropped connections')
    STATE.update(put_fail=0, puts=0)
    try:
        reuse_io.upload(base + '/big', out); ok(False, 'PUT 413 must be final')
    except reuse_io.TransferError as e:
        ok(e.code == 'result_http_413' and STATE['puts'] == 1, 'PUT 413 is final after one try: ' + e.code)
    STATE.update(put_fail=99, put_mode='503', puts=0)
    t = time.time()
    try:
        reuse_io.upload(base + '/res', out, deadline=time.time() + 0.001); ok(False, 'past the deadline must give up')
    except reuse_io.TransferError as e:
        ok(e.code == 'result_failed' and time.time() - t < 5, 'gives up at the deadline instead of retrying forever')

srv.shutdown()
print('io self-test: %s' % ('ALL OK' if not fails else '%d FAILED' % fails))
raise SystemExit(1 if fails else 0)
