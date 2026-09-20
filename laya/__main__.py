"""`python -m laya serve ...` and `laya serve ...`."""
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "serve":
        from .serving import main as serve
        return serve(argv[1:])
    if argv and argv[0] == "export-onnx":
        from .onnx_backend import main as export
        return export(argv[1:])
    print("usage: laya serve [--host H] [--port P] [--models a,b] [--device D]\n"
          "       laya export-onnx <checkpoint> <out_dir> [--quantize]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
