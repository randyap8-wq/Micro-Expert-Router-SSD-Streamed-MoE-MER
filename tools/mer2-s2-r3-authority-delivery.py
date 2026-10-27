#!/usr/bin/env python3
"""Private, one-shot delivery of *unapproved* S2-R3 authority evidence collector.

This launcher does not run the frozen MER test runner or grant test authority.
"""
import ast
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile

EXPECTED_ZIP = 'dfc15f2820fee91f5062713c79c4842660f867ea6335bd8256218c61678e99c3'
EXPECTED_FILES = {
  'mer2-s2-r3-authority-collect.py':'dc72c26e14557b7158256aad376995781125bb8d8761e5faa38df30f5af792d9',
  'MER2_S2_R3_Notion_Receipt_20261027.zip':'dd67eeef96a40ff08993e8c7234bbabde76cf58f3ad4e53ee40b1c3e5a3ca416',
}

def stop(message):
    raise SystemExit('STOP_S2_R3_PRIVATE_AUTHORITY_DELIVERY: '+message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def main():
    if os.geteuid()==0 or os.getuid()!=__import__('pwd').getpwnam('ubuntu').pw_uid:
        stop('expected ubuntu operator')
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        stop('expected Linux x86_64')
    if not Path('/usr/bin/python3.11').is_file():
        stop('Python 3.11 missing')
    parent=Path.home()/'mer-tools'
    if not parent.is_dir(): stop('mer-tools directory missing')
    os.umask(0o077)
    print('MER2 S2-R3: stage PRIVATE draft-authority collector (no MER execution).',flush=True)
    print('Paste the one-hour S3 presigned HTTPS URL for the new 9 KB authority bundle.',flush=True)
    url=getpass.getpass('S3 URL (hidden): ')
    parsed=urllib.parse.urlparse(url)
    host=parsed.hostname or ''
    allowed=host=='s3.amazonaws.com' or (host.startswith('s3.') and host.endswith('.amazonaws.com')) or (host.startswith('mer-amalgafy.s3.') and host.endswith('.amazonaws.com'))
    if not (parsed.scheme=='https' and allowed and not parsed.username and not parsed.password):
        stop('expected private Amazon S3 HTTPS URL')
    if not host.startswith('mer-amalgafy.s3.') and not parsed.path.startswith('/mer-amalgafy/'):
        stop('expected bucket mer-amalgafy')
    if 'X-Amz-Signature=' not in url:
        stop('presigned S3 query parameter missing')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):
            stop('unexpected S3 redirect')
    request=urllib.request.Request(url,headers={'User-Agent':'MER-S2-R3-private-authority-delivery/1.0'})
    opener=urllib.request.build_opener(NoRedirect)
    try:
        with opener.open(request,timeout=30) as response:
            if response.status!=200: stop('unexpected S3 status')
            archive_data=response.read(100_001)
    except urllib.error.HTTPError as e:
        stop('S3 HTTP '+str(e.code))
    except urllib.error.URLError:
        stop('S3 network error')
    finally:
        url=''
    if len(archive_data)>=100_001 or sha(archive_data)!=EXPECTED_ZIP:
        stop('private authority ZIP SHA256 mismatch')
    with zipfile.ZipFile(io.BytesIO(archive_data)) as z:
        if z.testzip() is not None: stop('authority ZIP CRC failed')
        if len(z.namelist())!=3 or set(z.namelist())!=set(EXPECTED_FILES)|{'MANIFEST.json'}:
            stop('unexpected authority ZIP members')
        manifest=json.loads(z.read('MANIFEST.json'))
        if manifest.get('execution_authorized') is not False:
            stop('bundle must be preparation-only')
        for name,expected in EXPECTED_FILES.items():
            if manifest.get(name,{}).get('sha256')!=expected or sha(z.read(name))!=expected:
                stop('authority bundle payload SHA mismatch: '+name)
        code=z.read('mer2-s2-r3-authority-collect.py')
        ast.parse(code.decode('utf-8'))  # syntax-only, never import/execute before verification
        root=Path(tempfile.mkdtemp(prefix='mer2-s2-r3-authority-delivery.',dir=parent))
        for name in EXPECTED_FILES:
            target=root/name
            with target.open('xb') as f:
                f.write(z.read(name)); f.flush(); os.fsync(f.fileno())
            target.chmod(0o600)
    print('S2_R3_AUTHORITY_DELIVERY=PASS_SHA256_AND_SYNTAX')
    print('AUTHORITY_DELIVERY_ROOT='+str(root))
    print('EXECUTION_AUTHORIZED=FALSE MER_BUILDS=0 MER_TESTS=0 MARKERS=0',flush=True)
    os.execv('/usr/bin/python3.11',['/usr/bin/python3.11','-B',str(root/'mer2-s2-r3-authority-collect.py')])

if __name__=='__main__':
    main()
