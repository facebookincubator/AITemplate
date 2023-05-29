{ pkgs ? import <nixpkgs> {
  config = {
    allowUnfree = true;
    cudaSupport = true;
  };
}}:

let 
  ait-deps = ps: with ps; [
    pytorch-bin
    pip
    wheel
    click
    unidecode
    inflect
    librosa
    jinja2
    sympy
    einops
    parameterized
    transformers
    # (
    #   buildPythonPackage rec {
    #     pname = "cuda_python";
    #     version = "12.1.0";
    #     format = "wheel";
    #     src = fetchPypi {
    #       inherit pname version format;
    #       sha256 = "94506d730baade1744767e2c05d5ddd84d7fbe4c9b6f694a54a3f376f7ffa525";
    #       abi = "cp39";
    #       python = "cp39";
    #       platform = "manylinux_2_17_x86_64.manylinux2014_x86_64";
    #     };
    #     doCheck = false;
    #   }
    # )
  ];  
in
pkgs.mkShell {
  buildInputs = [
    pkgs.cmake
    pkgs.cudatoolkit
    (pkgs.python310.withPackages ait-deps)
  ];

  shellHook = ''
    export CUDA_PATH=${pkgs.cudatoolkit}
    echo "You are now using a NIX environment"
  '';
}
