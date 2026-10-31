FROM docker.io/condaforge/miniforge3:latest

LABEL maintainer="DL-2026-02-L02"
LABEL description="Entorno de Deep Learning (PyTorch + CUDA + Conda) compatible con Distrobox y Dev Containers"

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# 1. Instalar dependencias del sistema y utilidades requeridas por Distrobox y Dev Containers
# Se cambia http:// por https:// en los repositorios de Ubuntu para evitar bloqueos de firewall/proxy HTTP
RUN rm -f /etc/dpkg/dpkg.cfg.d/excludes \
    && (sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true) \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        bash-completion \
        bc \
        bsdextrautils \
        bzip2 \
        ca-certificates \
        coreutils \
        curl \
        diffutils \
        findutils \
        git \
        gnupg \
        hostname \
        iproute2 \
        iputils-ping \
        less \
        libgl1 \
        libglib2.0-0 \
        locales \
        lsof \
        man-db \
        mount \
        openssh-client \
        passwd \
        pigz \
        pinentry-curses \
        procps \
        rsync \
        sudo \
        time \
        tree \
        tzdata \
        unzip \
        util-linux \
        wget \
        xauth \
        xz-utils \
        zip \
    && sed -i 's/^# *\(en_US.UTF-8\)/\1/' /etc/locale.gen \
    && locale-gen \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && touch /.containersetupdone

# 2. Configurar usuario no-root (UID 1000) con sudo sin contraseña para Dev Containers
RUN if id -u ubuntu >/dev/null 2>&1; then \
        usermod -aG sudo ubuntu; \
    else \
        useradd -m -u 1000 -s /bin/bash -G sudo ubuntu; \
    fi \
    && echo "%sudo ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/nopasswd \
    && chmod 0440 /etc/sudoers.d/nopasswd

# 3. Instalar compiladores C/C++ (necesarios para paquetes nativos y PyTorch/CUDA) y crear el entorno 'lab_pytorch'
RUN (sed -i 's|http://|https://|g' /etc/apt/sources.list.d/*.sources /etc/apt/sources.list 2>/dev/null || true) \
    && apt-get update \
    && apt-get install -y --no-install-recommends build-essential zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN mamba create -y -n lab_pytorch -c conda-forge python=3.10 pip pyreadstat \
    && mamba env update -n lab_pytorch -f /tmp/environment.yml \
    && mamba clean --all -f -y \
    && rm -f /tmp/environment.yml \
    && chmod -R a+rX /opt/conda

# 4. Activar el entorno 'lab_pytorch' a nivel de sistema (funciona en Distrobox con $HOME del host y en Dev Containers)
RUN printf '%s\n' \
    'export PATH="/opt/conda/envs/lab_pytorch/bin:/opt/conda/bin:$PATH"' \
    'if [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then' \
    '    . "/opt/conda/etc/profile.d/conda.sh"' \
    '    conda activate lab_pytorch 2>/dev/null || true' \
    'fi' \
    > /etc/profile.d/conda-lab.sh \
    && chmod +x /etc/profile.d/conda-lab.sh \
    && echo '[ -f /etc/profile.d/conda-lab.sh ] && . /etc/profile.d/conda-lab.sh' >> /etc/bash.bashrc

ENV CONDA_DEFAULT_ENV=lab_pytorch \
    CONDA_PREFIX=/opt/conda/envs/lab_pytorch \
    PATH="/opt/conda/envs/lab_pytorch/bin:/opt/conda/bin:${PATH}"

WORKDIR /workspaces/DL-2026-02-L02
CMD ["/bin/bash"]
