cd ./third_party

# 1. com_github_bazelbuild_buildtools
wget https://github.com/bazelbuild/buildtools/archive/refs/tags/4.2.2.tar.gz -O buildtools-4.2.2.tar.gz

# 2. bazel_skylib
wget https://github.com/bazelbuild/bazel-skylib/releases/download/1.3.0/bazel-skylib-1.3.0.tar.gz

# 3. rules_python
wget https://mirror.bazel.build/github.com/bazelbuild/rules_python/releases/download/0.4.0/rules_python-0.4.0.tar.gz

# 4. pybind11_bazel
wget https://github.com/pybind/pybind11_bazel/archive/26973c0ff320cb4b39e45bc3e4297b82bc3a6c09.zip -O pybind11_bazel-26973c0.zip

# 5. googletest
wget https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz -O googletest-1.12.1.tar.gz

# 6. rules_foreign_cc
wget https://github.com/bazelbuild/rules_foreign_cc/archive/0.5.1.tar.gz -O rules_foreign_cc-0.5.1.tar.gz

# 7. io_bazel_rules_docker
wget https://github.com/bazelbuild/rules_docker/releases/download/v0.23.0/rules_docker-v0.23.0.tar.gz

# 8. mlir-hlo
wget https://github.com/tensorflow/mlir-hlo/archive/ac26bdba7a5edfe6060ba5be528b9d20c987297d.zip -O mlir-hlo-ac26bdba.zip
