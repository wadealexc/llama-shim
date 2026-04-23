FROM node:22-alpine

WORKDIR /app

# Install git (needed by svelte.config.js for version resolution)
RUN apk add --no-cache git

# Install dependencies first (cached layer)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source (will be overridden by bind mount in compose)
COPY . .

# Change ownership to node user for runtime
RUN chown -R node:node /app

USER node

EXPOSE 5050

CMD ["npm", "run", "dev"]