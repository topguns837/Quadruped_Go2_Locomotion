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
    ISAACLAB_PATH=/workspace/isaaclab \
    GTK_A11Y=none \
    NO_AT_BRIDGE=1

SHELL ["/bin/bash", "-c"]
USER root

# System build dependencies required by Isaac Lab / Isaac Sim extensions.
# zenity is not a build dependency: Kit's own hang-detection watchdog shows a
# "Kit appears to be hanging, terminate?" dialog via zenity when its main
# thread stalls. Without it, that dialog can never display or be dismissed,
# so a stall becomes permanent limbo (observed directly: GPU memory pinned,
# 0% compute, black window forever) instead of a recoverable prompt.
# dbus-x11 (for dbus-launch) is zenity's own runtime dependency: without it,
# zenity's GTK init fails ("Unable to acquire session bus: Failed to execute
# child process 'dbus-launch'") and the dialog above crashes before it can
# ever display, recreating the exact permanent-limbo failure zenity was
# added to prevent (observed directly). Even with dbus-x11 installed, zenity
# then crashes a second way (Gtk-CRITICAL: unable to connect to the
# accessibility/at-spi bus, which isn't installed/running in this container
# either) — GTK_A11Y=none and NO_AT_BRIDGE=1 (set above) tell it to skip that
# entirely rather than pulling in yet another service just for a container-only
# recovery dialog nobody needs screen-reader support for (verified directly:
# zenity stays up waiting for input with these set, crashes without them).
#
# feh: viewer for scripts/live_dashboard.py's --output mode. That script needs a working matplotlib
# GUI backend to show a live window, but this image has none (no tkinter/Qt/GTK bindings in Isaac Sim's
# bundled Python) — installing PyQt5 to get one was tried and rejected: its bundled Qt build crashes on
# missing system XCB dependencies (`qt.qpa.plugin: Could not load the Qt platform plugin "xcb"`), and
# matplotlib auto-selects it as soon as it's importable even when only savefig() is used, not just
# plt.show(). feh sidesteps this entirely — a plain apt-installed binary correctly linked against this
# image's own system libraries (not a redistributable bundle), with `--reload <n>` to auto-refresh a
# PNG file on an interval. Do not add a Python Qt/GTK binding back to "fix" the dashboard window; the
# --output + feh combination is the working, deliberate solution.
#
# python3 + python3-tk: scripts/manual_command_slider.py's GUI (play.py's --manual_commands mode). Same
# constraint as feh above -- Isaac Sim's bundled Python (isaaclab -p) has no tkinter, and this is a
# *separate*, plain system Python installed specifically so that script can run outside that environment
# entirely, rather than trying to add a GUI toolkit into Isaac Sim's Python (the same dead end feh's
# comment already documents). python3-tk is the apt-packaged Tk binding, linked against this image's own
# system X11 libs like feh is -- not a pip wheel, so it doesn't hit the XCB problem either.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake dbus-x11 feh git libglib2.0-0 ncurses-term python3 python3-tk tmux wget zenity && \
    apt -y autoremove && apt clean autoclean && rm -rf /var/lib/apt/lists/*

# Fetch Isaac Lab pinned to the tag this project is compatible with
RUN git clone --branch ${ISAACLAB_VERSION} --depth 1 \
    https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}

RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh && \
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

# Pin the build-isolation setuptools version: a bare `isaaclab.sh --install`
# silently fails to install the core `isaaclab` package because one of its
# deps, flatdict==4.0.1, has a legacy setup.py that does `import
# pkg_resources` at build time, and a freshly-fetched setuptools no longer
# bundles it. Constraining setuptools keeps pkg_resources available for that
# build. See docker/README.md.
#
# pip<25 is pinned for two reasons, found together (both observed directly):
#   1. isaaclab.sh --install's own pip calls try to self-upgrade pip. On
#      Docker's OverlayFS, a pip invocation that replaces its own package
#      files while it is the process currently running and importing them
#      can corrupt its vendored submodules mid-write (observed: `pip's own
#      internal `import pip._vendor.packaging._structures` failing).
#      Constraining pip's version makes that self-upgrade attempt a no-op
#      ("already satisfies") instead of a live self-replace.
#   2. A newer pip (26.2.1, from an earlier unconstrained `pip install
#      --upgrade pip` here) does NOT apply PIP_CONSTRAINT inside the isolated
#      build environment it spins up for flatdict's legacy setup.py, so that
#      isolated env pulls an unconstrained fresh setuptools>=81 regardless of
#      the pin above, reproducing the exact same pkg_resources failure one
#      layer deeper. Verified directly: pip 26.2.1 fails building flatdict,
#      pip 24.2 builds it cleanly, both against the identical constraints
#      file. Pinning here keeps pip itself inside the version this whole
#      constraint scheme was verified against.
RUN echo "setuptools<81" > /etc/pip-constraints.txt && \
    echo "pip<25" >> /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install "pip==24.2"

# wheel/toml are this project's own build-backend requirements (see
# source/quadruped_go2_locomotion/pyproject.toml). Pre-installing them here so
# entrypoint.sh's `pip install -e` can run with --no-build-isolation means
# that install never needs a PyPI round-trip on every container start.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml wheel

# matplotlib/tensorboard for scripts/live_dashboard.py. Both already happen to be present as transitive
# dependencies pulled in by other isaaclab extensions, but installing them explicitly means the
# dashboard's own dependencies don't silently break if some unrelated package's requirements ever change.
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install matplotlib tensorboard

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
#
# libnvblas.so (shipped inside nvidia-cublas-cu12) is excluded outright, not
# just probed like the rest: preloading it makes NVBLAS intercept BLAS calls
# process-wide, and it has no CPU-BLAS fallback configured here (no
# nvblas.conf) — "[NVBLAS] CPU Blas library need to be provided" printed
# right before every play.py hang seen so far. It doesn't error, it hangs the
# process indefinitely the first time something hits a BLAS call NVBLAS can't
# service (near-0% GPU util, one CPU core pegged, forever). We don't need
# NVBLAS at all — torch's own CUDA acceleration doesn't route through it.
# Verified directly: removing this one entry from LD_PRELOAD is the
# difference between play.py hanging at actor/critic construction every time
# and running the simulation normally (GPU util 5%→41%, power 27W→111W),
# on an otherwise identical container.
RUN cuda_libs=$(for f in $(find ${ISAACSIM_ROOT_PATH}/kit/python/lib/python3.11/site-packages/nvidia -maxdepth 3 -name '*.so*' -not -path '*/cu13/*' -not -name 'libnvblas.so*' | sort); do if ! ldd "$f" 2>&1 | grep -q "not found"; then echo "$f"; fi; done | tr '\n' ':') && \
    sed -i "s|LD_PRELOAD=\$SCRIPT_DIR/kit/libcarb.so|LD_PRELOAD=\$SCRIPT_DIR/kit/libcarb.so:${cuda_libs%:}|" \
        ${ISAACSIM_ROOT_PATH}/python.sh

# ${ISAACSIM_ROOT_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/torch ships
# with this base image as vestigial packaging leftovers: no __init__.py, no
# compiled _C extension, nothing but a `_vendor/packaging` dir and a
# `share/cmake` dir (libtorch's C++ CMake config, not Python code at all). A
# bare `import torch` from a shell is unaffected (finds the real, complete
# torch in kit/python/site-packages first) but Kit's extension loader
# (omni.ext._impl.custom_importer) scopes sys.path per-extension, and at
# least one startup extension (isaacsim.gui.menu, observed directly) hits
# this directory before the real site-packages torch is reachable. Since it
# has no __init__.py, Python resolves it as an empty PEP 420 namespace
# package instead of erroring — so `import torch` "succeeds" but yields a
# module with nothing in it. That broken module then sits in sys.modules for
# the rest of the process, so every later `import torch` anywhere (isaaclab,
# isaaclab_assets, isaaclab_tasks, tensordict, ...) reuses the same broken
# object: `AttributeError: module 'torch' has no attribute 'Tensor'`,
# cascading through every extension that touches torch. Deleting this
# directory removes it from consideration entirely, so every `import torch`
# — from a shell or from inside Kit's extension loader — resolves to the one
# real, complete torch install. Verified directly: play.py's extension chain
# (isaaclab_assets, isaaclab_tasks, isaaclab_rl, isaaclab_mimic) loads
# cleanly after this removal, reproduces the AttributeError cascade without
# it, on an otherwise identical container.
RUN rm -rf ${ISAACSIM_ROOT_PATH}/exts/omni.isaac.ml_archive/pip_prebundle/torch

# unitree_sdk2py: real-hardware deployment (deploy/deploy_real.py, see deploy.md). Not on PyPI at all
# (confirmed directly: `pip install unitree_sdk2py` finds no matching distribution) -- installed from its
# GitHub source, matching how Unitree themselves document installing it.
#
# Pinned to a specific commit (not a branch/tag) -- this project has no control over unitree_sdk2py's
# upstream, and an unpinned `git+https://.../unitree_sdk2_python.git` would silently pick up whatever
# is on master at image-build time, risking a future rebuild breaking (API changes, or re-hitting/regressing
# the exact packaging gaps documented below in ways not yet seen). ${UNITREE_SDK2PY_COMMIT} is defined once
# and reused for both this install and the .so-file fix further down so they can never drift apart from
# each other.
ARG UNITREE_SDK2PY_COMMIT=65691c8a8bc53b98d3976dba4dbf9d5d20b2e7f5
#
# unitree_sdk2py's own dependency, cyclonedds==0.10.2 (the Python DDS bindings), also has no prebuilt wheel for this
# platform and needs the native CycloneDDS C library present to build against (`Could not locate
# cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH`, confirmed directly) -- there's no apt
# package for it either (`apt-cache search cyclonedds` returns nothing in this image). build-essential and
# cmake are already installed above (line ~76), so no new apt packages are needed, just building this one
# library from source and pointing pip at it via CYCLONEDDS_HOME.
RUN git clone --branch releases/0.10.x --depth 1 https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds && \
    cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF && \
    cmake --build /tmp/cyclonedds/build -j$(nproc) && \
    cmake --install /tmp/cyclonedds/build && \
    ldconfig && \
    rm -rf /tmp/cyclonedds

# unitree_sdk2py's install silently upgrades numpy to 2.x as a transitive dependency (observed directly:
# pip pulled numpy-2.4.6, no version ceiling declared anywhere in its own requirements) -- which then
# breaks isaaclab/isaaclab-rl/isaaclab-tasks, all of which pin numpy<2 (confirmed directly: pip's own
# dependency-conflict warnings named all three after this install, and would have broken every future
# `isaaclab -p` launch in this image, though an already-running process is unaffected since Python doesn't
# hot-reload site-packages). The `pip install "numpy<2"` immediately after re-pins it back down; verified
# directly afterward that unitree_sdk2py's own imports (ChannelFactoryInitialize, LowCmd_/LowState_/
# SportModeState_, CRC, RecurrentThread, SportClient, MotionSwitcherClient) all still work fine against the
# older numpy -- nothing in the actual code paths this project uses needs 2.x. opencv-python (also pulled
# in transitively, unused by deploy_real.py, and left version-mismatched by this repin) is harmless as long
# as nothing imports cv2 -- deploy_real.py doesn't.
RUN CYCLONEDDS_HOME=/usr/local CMAKE_PREFIX_PATH=/usr/local \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@${UNITREE_SDK2PY_COMMIT}" && \
    ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install "numpy<2"

# unitree_sdk2py.utils.crc.CRC (used to stamp every outgoing LowCmd -- required, not optional) loads a
# prebuilt native library, crc_amd64.so, via ctypes at runtime. That file exists in the upstream source
# repo (unitree_sdk2py/utils/lib/crc_amd64.so) but the pip install above silently drops it -- confirmed
# directly: `CRC()` raised `OSError: .../utils/lib/crc_amd64.so: cannot open shared object file`, and the
# installed package has no `utils/lib/` directory at all. This is an upstream packaging gap (the .so isn't
# declared as package data for a PEP 517 wheel build), not something fixable via pip flags -- so fetch the
# exact same pinned commit again just for this one directory and copy it into the already-installed package
# by hand. Verified directly afterward: CRC().Crc(...) computes successfully.
RUN git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git /tmp/unitree_sdk2_python_src && \
    git -C /tmp/unitree_sdk2_python_src fetch --depth 1 origin ${UNITREE_SDK2PY_COMMIT} && \
    git -C /tmp/unitree_sdk2_python_src checkout ${UNITREE_SDK2PY_COMMIT} && \
    UNITREE_SDK2PY_DIR=$(${ISAACLAB_PATH}/_isaac_sim/python.sh -c "import unitree_sdk2py, os; print(os.path.dirname(unitree_sdk2py.__file__))") && \
    mkdir -p ${UNITREE_SDK2PY_DIR}/utils/lib && \
    cp /tmp/unitree_sdk2_python_src/unitree_sdk2py/utils/lib/*.so ${UNITREE_SDK2PY_DIR}/utils/lib/ && \
    rm -rf /tmp/unitree_sdk2_python_src

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
