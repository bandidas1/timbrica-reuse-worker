# Moving audio between our server and the worker — pure Python (no torch), so the self-test runs in CI.
#
# Contract (same shape as the voice-cover lane, App\Services\VoiceCover\SvcTransfer): one signed GET for the
# source, one signed PUT for the result; the PUT is idempotent on our side (last valid write wins), so retrying
# it is always safe. What retries and what does not:
#   - a dropped connection, a timeout, 408 / 425 / 429 / 5xx  -> retry with backoff until the deadline;
#   - 403 / 404 / 410 / 413 / 422 (bad signature, gone job, too big, not audio) -> final, no retry: repeating
#     it can only waste the worker's paid seconds.
import os
import time

import requests

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}
FINAL_STATUS = {400, 401, 403, 404, 409, 410, 411, 413, 415, 422}
CHUNK = 1 << 20


class TransferError(Exception):
    def __init__(self, code, detail=''):
        super().__init__(code + (': ' + detail if detail else ''))
        self.code = code


def _sleep_for(attempt):
    return min(30.0, 2.0 * (2 ** attempt))


def download(url, dst, deadline=0.0, timeout=60, max_attempts=6, session=None):
    """Stream the source to dst; returns its size. Retries the transient failures above."""
    s = session or requests
    attempt = 0
    while True:
        try:
            with s.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in FINAL_STATUS:
                    raise TransferError('source_http_%d' % r.status_code)
                if r.status_code != 200:
                    raise _Retry('source_http_%d' % r.status_code)
                n = 0
                with open(dst, 'wb') as f:
                    for buf in r.iter_content(CHUNK):
                        if buf:
                            f.write(buf)
                            n += len(buf)
                if n == 0:
                    raise _Retry('source_empty')
                return n
        except TransferError:
            raise
        except (_Retry, requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            attempt += 1
            if attempt >= max_attempts or (deadline and time.time() + _sleep_for(attempt) > deadline):
                raise TransferError('source_failed', str(e)[:120])
            time.sleep(_sleep_for(attempt))


def upload(url, path, content_type='audio/flac', deadline=0.0, timeout=120, max_attempts=8, session=None):
    """PUT the result file; returns the HTTP status of the accepted attempt."""
    s = session or requests
    size = os.path.getsize(path)
    attempt = 0
    while True:
        try:
            with open(path, 'rb') as f:
                r = s.put(url, data=f, timeout=timeout,
                          headers={'Content-Type': content_type, 'Content-Length': str(size)})
            if r.status_code in (200, 201, 204):
                return r.status_code
            if r.status_code in FINAL_STATUS:
                raise TransferError('result_http_%d' % r.status_code, (r.text or '')[:120])
            raise _Retry('result_http_%d' % r.status_code)
        except TransferError:
            raise
        except (_Retry, requests.ConnectionError, requests.Timeout) as e:
            attempt += 1
            if attempt >= max_attempts or (deadline and time.time() + _sleep_for(attempt) > deadline):
                raise TransferError('result_failed', str(e)[:120])
            time.sleep(_sleep_for(attempt))


class _Retry(Exception):
    pass
