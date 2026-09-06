# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Two-stage build:
#   base: everything expensive and one-time (apt packages, Isaac Lab source,
#         every pip install including isaaclab.sh --install). Only changes
#         when the Isaac Sim/Isaac Lab version pins change.
#   app:  cheap, fast-to-rebuild bits on top (env/bashrc, entrypoint). This
#         is the stage docker-compose.yml actually builds/runs.
#
# Neither stage copies this repository's source code, scripts, or USD
# assets: those are bind-mounted at `docker run` time (see
# ../docker-compose.yml) so editing project code never requires a rebuild.

ARG ISAACSIM_BASE_IMAGE=nvcr.io/nvidia/isaac-sim
ARG ISAACSIM_VERSION=5.1.0

# ---------------------------------------------------------------------------
# Stage: base
# ---------------------------------------------------------------------------
FROM ${ISAACSIM_BASE_IMAGE}:${ISAACSIM_VERSION} AS base

# Isaac Lab version this project is developed/tested against (see README.md)
ARG ISAACLAB_VERSION=v2.3.2

ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    ACCEPT_EULA=Y \
    OMNI_KIT_ALLOW_ROOT=1 \
    ISAACSIM_ROOT_PATH=/isaac-sim \
    ISAACLAB_PATH=/workspace/isaaclab

SHELL ["/bin/bash", "-c"]
USER root

# System build dependencies required by Isaac Lab / Isaac Sim extensions.
# zenity is not a build dependency: Kit's own hang-detection watchdog shows a
# "Kit appears to be hanging, terminate?" dialog via zenity when its main
# thread stalls. Without it, that dialog can never display or be dismissed,
# so a stall becomes permanent limbo (observed directly: GPU memory pinned,
# 0% compute, black window forever) instead of a recoverable prompt.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libglib2.0-0 ncurses-term tmux wget zenity && \
    apt -y autoremove && apt clean autoclean && rm -rf /var/lib/apt/lists/*

# Fetch Isaac Lab pinned to the tag this project is compatible with
RUN git clone --branch ${ISAACLAB_VERSION} --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh && \
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

# apt-based dependencies declared by individual Isaac Lab extensions
RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && rm -rf /var/lib/apt/lists/*

# Cache/log directories expected by Isaac Sim (also required for Singularity/Apptainer portability)
RUN mkdir -p ${ISAACSIM_ROOT_PATH}/kit/cache \
    /root/.cache/ov /root/.cache/pip /root/.cache/nvidia/GLCache \
    /root/.nv/ComputeCache /root/.nvidia-omniverse/logs \
    /root/.local/share/ov/data /root/Documents

# NVIDIA binary placeholders (overridden by the nvidia container runtime at `docker run`)
RUN touch /bin/nvidia-smi /bin/nvidia-debugdump /bin/nvidia-persistenced \
    /bin/nvidia-cuda-mps-control /bin/nvidia-cuda-mps-server /etc/localtime && \
    mkdir -p /var/run/nvidia-persistenced && touch /var/run/nvidia-persistenced/socket

# Pin the build-isolation setuptools version: a bare `isaaclab.sh --install`
# silently fails to install the core `isaaclab` package because one of its
# deps, flatdict==4.0.1, has a legacy setup.py that does `import
# pkg_resources` at build time, and a freshly-fetched setuptools no longer
# bundles it. Constraining setuptools keeps pkg_resources available for that
# build. See docker/README.md.
RUN echo "setuptools<81" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

# isaaclab.sh --install pulls several 500MB+ wheels (torch, nvidia-cudnn-cu13,
# etc.) from PyPI. pip's default read timeout (15s per chunk) is too tight on
# a slow or unstable connection, and this connection has repeatedly stalled
# hard enough (observed on different large packages each time, after 25-55+
# minutes of otherwise-successful progress) that even a raised timeout alone
# wasn't enough. Combine a longer timeout with a retry loop around the whole
# command: the pip cache mount below persists what's already downloaded
# across attempts, so a retry only has to push through whatever package was
# mid-download when the connection stalled, not start over.
ENV PIP_DEFAULT_TIMEOUT=180

# Install Isaac Lab's own Python dependencies (rsl-rl, isaaclab, etc.)
RUN --mount=type=cache,target=/root/.cache/pip \
    for i in 1 2 3 4 5; do \
        ${ISAACLAB_PATH}/isaaclab.sh --install && break; \
        echo "isaaclab.sh --install failed (attempt $i/5), retrying..." >&2; \
        if [ "$i" = 5 ]; then exit 1; fi; \
        sleep 10; \
    done

# Matches upstream Isaac Lab's own Dockerfile.base, which removes quadprog
# after install.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog

# warp-lang (an isaaclab dependency) pulls in newer, CUDA-13-targeted nvidia-*
# packages. For some of torch's actual cu12 dependencies this displaces the
# cu12 build entirely (observed with both nvidia-cusparselt-cu12 and
# nvidia-cufile-cu12, each time as a different `ImportError:
# lib*.so: cannot open shared object file` when importing torch, one library
# at a time as different extensions happened to exercise it). Force-reinstall
# every nvidia-*-cu12 package torch itself declares as a dependency (see
# `pip show torch`), not just the ones seen failing so far.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --force-reinstall --no-deps \
    nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 \
    nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 \
    nvidia-cufile-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 \
    nvidia-cusparse-cu12 nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
    nvidia-nvjitlink-cu12 nvidia-nvtx-cu12

# Even with the correct files in place, Kit's extension loader
# (omni.ext._impl.custom_importer) fails to resolve these via torch's normal
# $ORIGIN-relative RPATH lookup, while a plain script-level `import torch`
# works fine, so preloading is required to close that gap.
#
# Setting LD_PRELOAD as an image ENV does NOT work here: _isaac_sim/python.sh
# (the actual launcher every `isaaclab.sh -p` invocation runs) unconditionally
# does `export LD_PRELOAD=$SCRIPT_DIR/kit/libcarb.so` as its own workaround
# for a missing-library issue, silently overwriting (not appending to)
# whatever LD_PRELOAD was already set. So patch that script directly, and
# preload every .so shipped by the cu12 packages above (found dynamically
# rather than hardcoded, since several of them, e.g. cudnn, ship many files)
# instead of playing whack-a-mole one library at a time.
#
# A blanket "preload every .so found" is NOT safe: some of these ship
# optional extras with their own transitive dependencies on real hardware
# support libraries this container doesn't have (observed directly, one at a
# time: libcufile_rdma.so.1 needs libmlx5.so.1, an InfiniBand/RDMA library;
# some NCCL build needs libmpi.so.40, an Open MPI library). LD_PRELOAD
# failures abort the ENTIRE process before any Python code runs, so
# preloading even one such library broke every python invocation in the
# container outright, not just torch (observed: container exited immediately
# with "error while loading shared libraries"). So instead of guessing which
# filename patterns to exclude, actually try to load each candidate at build
# time and only preload the ones that succeed in this exact environment.
RUN cuda_libs=$(for f in $(find ${ISAACSIM_ROOT_PATH}/kit/python/lib/python3.11/site-packages/nvidia -maxdepth 3 -name '*.so*' -not -path '*/cu13/*' | sort); do if ! ldd "$f" 2>&1 | grep -q "not found"; then echo "$f"; fi; done | tr '\n' ':') && \
    sed -i "s|LD_PRELOAD=\$SCRIPT_DIR/kit/libcarb.so|LD_PRELOAD=\$SCRIPT_DIR/kit/libcarb.so:${cuda_libs%:}|" \
        ${ISAACSIM_ROOT_PATH}/python.sh

# ---------------------------------------------------------------------------
# Stage: app
# ---------------------------------------------------------------------------
FROM base AS app

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> /root/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> /root/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> /root/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> /root/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc

# This is where docker-compose.yml bind-mounts the repository at runtime
ENV PROJECT_PATH=/workspace/quadruped_go2_locomotion
WORKDIR ${PROJECT_PATH}

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
