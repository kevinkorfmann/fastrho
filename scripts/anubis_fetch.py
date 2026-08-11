"""Fetch a file from behind an Anubis (within.website) proof-of-work gate.

Anubis embeds a challenge in the page; the browser brute-forces a SHA-256 nonce, then calls
pass-challenge to receive an auth cookie. We reproduce that headlessly. Usage:
    python anubis_fetch.py <url> <out_path>
"""
import sys, re, json, time, hashlib, http.cookiejar, urllib.request, urllib.parse

URL = sys.argv[1]
OUT = sys.argv[2]
UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

cj = http.cookiejar.CookieJar()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def opener(follow=True):
    handlers = [urllib.request.HTTPCookieProcessor(cj)]
    if not follow:
        handlers.append(NoRedirect())
    op = urllib.request.build_opener(*handlers)
    op.addheaders = [("User-Agent", UA), ("Accept", "text/html,*/*")]
    return op


def get(url, follow=True, data=None):
    try:
        r = opener(follow).open(urllib.request.Request(url, data=data), timeout=90)
        return r.getcode(), r.read(), dict(r.headers), r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers), url


def script_json(sid, h):
    m = re.search(r'<script id="%s" type="application/json">(.*?)</script>' % sid, h, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except Exception:
        return None


# 1. load the challenge page (retry: Anubis 500s / issues null challenges intermittently)
html, chal = None, None
for attempt in range(10):
    code, body, hdr, _ = get(URL)
    html = body.decode("utf-8", "replace")
    print(f"[1.{attempt}] challenge page HTTP {code}, {len(html)} bytes, cookies {[c.name for c in cj]}")
    if code == 200 and "anubis_challenge" not in html:
        print("[!] no challenge present; treating response as the file")
        open(OUT, "wb").write(body)
        sys.exit(0)
    chal = script_json("anubis_challenge", html)
    if chal and isinstance(chal, dict) and chal.get("challenge"):
        break
    time.sleep(7)
if not (chal and isinstance(chal, dict) and chal.get("challenge")):
    print("[x] could not obtain a valid challenge after retries")
    sys.exit(2)
base = script_json("anubis_base_prefix", html) or ""
ver = script_json("anubis_version", html)
print(f"[2] version={ver} base_prefix={base!r}")
print(f"    challenge JSON keys: {list(chal) if isinstance(chal, dict) else type(chal)}")
print(f"    challenge JSON: {json.dumps(chal)[:400]}")

# Anubis 1.24 "fast": hash challenge.randomData + nonce (main.mjs passes p.randomData as data)
cobj = chal["challenge"]
data = cobj["randomData"]
cid = cobj.get("id")
rules = chal.get("rules", {})
difficulty = int(rules.get("difficulty", 4))
algorithm = rules.get("algorithm", "fast")
print(f"[3] solving PoW: difficulty={difficulty} algorithm={algorithm} id={cid}")

# 2. brute-force nonce: sha256(randomData + nonce) hex has `difficulty` leading zero nibbles
t0 = time.time()
prefix = "0" * difficulty
nonce = 0
while True:
    h = hashlib.sha256((data + str(nonce)).encode()).hexdigest()
    if h.startswith(prefix):
        break
    nonce += 1
    if nonce % 5_000_000 == 0:
        print(f"    ...{nonce} tried")
elapsed_ms = int((time.time() - t0) * 1000)
print(f"[4] solved: nonce={nonce} hash={h[:16]}... in {elapsed_ms} ms")

# 3. pass-challenge -> sets the auth cookie (do not follow the redirect to the big file)
params = {"response": h, "nonce": str(nonce), "redir": URL, "elapsedTime": str(max(elapsed_ms, 1))}
if cid:
    params["id"] = cid
pass_url = f"{base}/.within.website/x/cmd/anubis/api/pass-challenge?" + urllib.parse.urlencode(params)
if pass_url.startswith("/"):
    pass_url = "https://" + urllib.parse.urlparse(URL).netloc + pass_url
code, pbody, phdr, _ = get(pass_url, follow=False)
print(f"[5] pass-challenge: HTTP {code}; cookies now: {[c.name for c in cj]}")
print(f"    Location: {phdr.get('Location','(none)')}")
if code not in (200, 302, 307):
    print("    body:", pbody[:300])

# 4. stream the file with the auth cookie
req = urllib.request.Request(URL, headers={"User-Agent": UA})
r = opener(True).open(req, timeout=120)
ct = r.headers.get("Content-Type", "")
cl = r.headers.get("Content-Length", "?")
print(f"[6] download: HTTP {r.getcode()} content-type={ct} content-length={cl}")
if "text/html" in ct:
    peek = r.read(300)
    print("    STILL BLOCKED (got HTML):", peek[:200])
    sys.exit(1)
n = 0
with open(OUT, "wb") as f:
    while True:
        chunk = r.read(1 << 20)
        if not chunk:
            break
        f.write(chunk)
        n += len(chunk)
        if n % (50 << 20) < (1 << 20):
            print(f"    ...{n/1e6:.0f} MB")
print(f"[7] DONE: wrote {n} bytes -> {OUT}")
