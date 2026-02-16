FROM node:18-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --default-timeout=120 sympy antlr4-python3-runtime==4.11.1

WORKDIR /workspace

COPY package.json package-lock.json ./
RUN npm install

COPY . .

RUN npm run compile

CMD ["node", "out/test/runTest.js"]
