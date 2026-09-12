import os, hashlib, urllib.request, concurrent.futures as cf, time
D = "/content/fly/data"; os.makedirs(D, exist_ok=True)
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
 "annotations.feather":      "body-annotations-male-cns-v1.0-minconf-0.5.feather",
 "neurotransmitters.feather":"body-neurotransmitters-male-cns-v1.0.feather",
 "edges.feather":            "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
def fetch(item):
    name, fn = item
    p = os.path.join(D, name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return name, os.path.getsize(p), "skip"
    t0 = time.time()
    urllib.request.urlretrieve(BASE + fn, p)
    return name, os.path.getsize(p), "%.1fs" % (time.time() - t0)

with cf.ThreadPoolExecutor(3) as ex:
    for name, size, how in ex.map(fetch, FILES.items()):
        print("%-26s %10.1f MB  %s" % (name, size/1e6, how))

print()
for name in FILES:
    p = os.path.join(D, name)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    print("%-26s sha256=%s" % (name, h.hexdigest()[:32]))
print("\n总占用 MB:", round(sum(os.path.getsize(os.path.join(D,n)) for n in FILES)/1e6, 1))
