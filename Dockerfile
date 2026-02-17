FROM node:18-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir --default-timeout=120 \
    sympy==1.14.0 antlr4-python3-runtime==4.11.1 pytest==8.3.4

WORKDIR /workspace

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

RUN npm run compile

RUN useradd -m appuser
USER appuser

CMD ["node", "out/test/runTest.js"]
