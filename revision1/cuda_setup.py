import os, sys
base = os.path.join(sys.prefix, "Lib", "site-packages", "nvidia")
for pkg in ["cublas/bin", "cudnn/bin"]:
    p = os.path.join(base, *pkg.split("/"))
    os.environ["PATH"] = p + os.pathsep + os.environ["PATH"]
    os.add_dll_directory(p)