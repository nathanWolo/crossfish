#!/usr/bin/env python3
"""Pack a native binary into a CodinGame Python 3 submission.

  python3 tools/cg_native/pack.py <binary> -o cpp_impl/cg_input_native.py
      [--manifest tools/cg_native/manifest.json --source FILE ... --note KEY=VALUE ...]

tools/cg_native/build.sh calls this; see documentation/native_build.md.

The binary is xz-compressed (x86 BCJ filter + LZMA2 preset 9e) and written into
one string literal in the U15 alphabet (tools/nnue_cjk14.py): 15 bits per
character, U+3400..U+9FFF then U+E000..U+F3FF, all single UTF-16 units, no
surrogates, nothing an editor normalizes. CodinGame counts the 100,000-character
cap in UTF-16 units, so len(text) is the counted size.

The launcher (stdlib only) checks the dynamic binary's runtime (glibc >= 2.34,
libstdc++.so.6; a clear stderr line if not), decodes, writes the binary to an
anonymous memfd (fallbacks: /tmp, the working directory, the script's
directory), and os.execv's it, so the engine inherits stdin/stdout directly and
Python is gone before the first read. Arguments pass through, so
`python3 cg_input_native.py selfcheck 120 9` and `... match` work.

The output depends only on the binary's bytes (and on liblzma's encoder, which
the manifest pins by recording the payload's own hash), so packing the same
binary twice gives the same file.
"""
import argparse
import hashlib
import json
import lzma
import sys
from pathlib import Path

SPLIT = 0xA000 - 0x3400  # 27,648 characters in the first range
CAP = 100000             # CodinGame's limit, in UTF-16 units


def u15(data: bytes) -> str:
    b = "".join(format(x, "08b") for x in data)
    b += "0" * ((-len(b)) % 15)
    vals = (int(b[i:i + 15], 2) for i in range(0, len(b), 15))
    return "".join(chr(0x3400 + v) if v < SPLIT else chr(0xE000 + v - SPLIT) for v in vals)


def u15_decode(text: str, nbytes: int) -> bytes:
    """The launcher's own decode, for tests: U15 characters back to nbytes bytes."""
    b = "".join(format(o - 13312 if o < 40960 else o - 29696, "015b") for o in map(ord, text))
    return int(b[:8 * nbytes], 2).to_bytes(nbytes, "big")


def xz_compress(raw: bytes) -> bytes:
    return lzma.compress(raw, format=lzma.FORMAT_XZ, check=lzma.CHECK_NONE, filters=[
        {"id": lzma.FILTER_X86},
        {"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": 3, "lp": 0, "pb": 0},
    ])


LAUNCHER = '''import os,sys,lzma
D="{payload}"
def E(m):sys.stderr.write("crossfish launcher: %s\\n"%m);sys.stderr.flush()
def main():
 b="".join(format(o-13312 if o<40960 else o-29696,"015b")for o in map(ord,D))
 x=lzma.decompress(int(b[:{nbits}],2).to_bytes({nbytes},"big"))
 a=sys.argv[1:]
 try:
  v=os.confstr("CS_GNU_LIBC_VERSION");import ctypes;ctypes.CDLL("libstdc++.so.6")
  if tuple(map(int,v.split()[1].split(".")[:2]))<(2,34):E("glibc too old for this binary: "+v)
 except Exception as e:E("preflight: %r"%e)
 try:
  f=os.memfd_create("cf",0);os.write(f,x);os.execv("/proc/self/fd/%d"%f,["cf"]+a)
 except Exception as e:E("memfd: %r"%e)
 for d in("/tmp",os.getcwd(),os.path.dirname(os.path.abspath(__file__))):
  p=os.path.join(d,"cf_{tag}")
  try:
   if not(os.path.exists(p)and os.path.getsize(p)==len(x)):
    t="%s.%d"%(p,os.getpid());f=os.open(t,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o755);os.write(f,x);os.close(f);os.chmod(t,0o755);os.replace(t,p)
   os.execv(p,[p]+a)
  except Exception as e:E("%s: %r"%(p,e))
 E("could not exec the engine");sys.exit(1)
main()
'''


def render(raw: bytes, xz: bytes) -> str:
    tag = hashlib.sha256(raw).hexdigest()[:12]
    return LAUNCHER.format(payload=u15(xz), nbits=8 * len(xz), nbytes=len(xz), tag=tag)


def source_sha256(path: Path) -> str:
    """sha256 with CRLF normalized to LF, so Windows checkouts hash like Linux ones."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--manifest", help="write the build record (JSON) here")
    ap.add_argument("--root", default=".", help="repository root; manifest paths are relative to it")
    ap.add_argument("--source", action="append", default=[], help="a source file the binary was built from")
    ap.add_argument("--note", action="append", default=[], help="KEY=VALUE recorded in the manifest")
    args = ap.parse_args()
    raw = Path(args.binary).read_bytes()
    xz = xz_compress(raw)
    assert lzma.decompress(xz) == raw
    text = render(raw, xz)
    assert not any(0xD800 <= ord(c) <= 0xDFFF for c in text)
    out = Path(args.out)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    units = len(text.encode("utf-16-le")) // 2
    payload = len(u15(xz))
    print(f"binary {len(raw):,} B sha256 {hashlib.sha256(raw).hexdigest()}")
    print(f"xz {len(xz):,} B ({len(xz) / len(raw):.3f})")
    print(f"payload {payload:,} chars; file {units:,} UTF-16 units "
          f"({units - payload:,} launcher), {len(text.encode()):,} UTF-8 bytes; "
          f"headroom {CAP - units:,} units to the {CAP:,} cap")
    print(f"{out} sha256 {hashlib.sha256(text.encode()).hexdigest()}")
    if args.manifest:
        root = Path(args.root).resolve()
        rel = lambda p: Path(p).resolve().relative_to(root).as_posix()
        manifest = {
            "submission": rel(out),
            "submission_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "submission_utf16_units": units,
            "binary_sha256": hashlib.sha256(raw).hexdigest(),
            "binary_bytes": len(raw),
            "xz_bytes": len(xz),
            "xz_sha256": hashlib.sha256(xz).hexdigest(),
        }
        for kv in args.note:
            k, _, v = kv.partition("=")
            manifest[k] = v
        manifest["sources"] = {rel(s): source_sha256(Path(s)) for s in args.source}
        with open(args.manifest, "w", encoding="utf-8", newline="\n") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"manifest {args.manifest}")
    if units >= CAP:
        sys.exit("OVER THE CAP")


if __name__ == "__main__":
    main()
