"""`python -m laya <command> ...` and `laya <command> ...`."""
import sys

USAGE = """usage: laya serve [--host H] [--port P] [--models a,b] [--device D]
       laya export-onnx <checkpoint> <out_dir> [--quantize]
       laya prepare-data typed-decisions|open-jev --out DIR [--config NAME]
       laya train --data FILE --base NAME|DIR --out DIR [options; see laya train --help]
       laya eval NAME|DIR --data FILE [--device D] [--out report.json]"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd == "serve":
        from .serving import main as run
    elif cmd == "export-onnx":
        from .onnx_backend import main as run
    elif cmd == "prepare-data":
        from .data import main as run
    elif cmd == "train":
        from .train import main as run
    elif cmd == "eval":
        from .evaluate import main as run
    else:
        print(USAGE, file=sys.stderr)
        return 2
    return run(rest)


if __name__ == "__main__":
    sys.exit(main())
