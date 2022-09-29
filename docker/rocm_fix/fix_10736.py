src = ""
with open("/opt/rocm/hip/bin/hipcc.pl", "r") as fi:
    src = fi.read()

src = src.replace(
    "$HIP_CLANG_TARGET = chomp($HIP_CLANG_TARGET);", "chomp($HIP_CLANG_TARGET);"
)
with open("/opt/rocm/hip/bin/hipcc.pl", "w") as fo:
    fo.write(src)
