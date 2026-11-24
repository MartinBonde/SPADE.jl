# Pkg runs this file when it installs SPADE. It downloads the compiled solver from the
# release on GitHub. The artifact is lazy, so that Pkg does not first ask the Julia
# package server, which does not have the solver, and print a failure.
using Pkg.Artifacts: artifact_meta, download_artifact

meta = artifact_meta("spade", joinpath(dirname(@__DIR__), "Artifacts.toml"))
# Other systems have no solver, and the first solve says so.
if meta !== nothing
    source = only(meta["download"])
    hash = Base.SHA1(meta["git-tree-sha1"])
    result = download_artifact(hash, source["url"], source["sha256"])
    result === true || throw(result)
end
