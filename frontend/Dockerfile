FROM node:20-slim
WORKDIR /app
COPY package.json package.json
RUN npm install
COPY astro.config.mjs astro.config.mjs
COPY src src
ENV VITE_API_URL=http://localhost:8000
EXPOSE 4321
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD node -e "fetch('http://localhost:4321').then(()=>process.exit(0)).catch(()=>process.exit(1))"
CMD ["npm", "run", "dev"]
