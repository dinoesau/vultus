# Secrets - donde sacar cada valor

Solo lectura local usa `.env.example`.
Todo prod vive en GitHub Environment `production` y en `modal secret`, nunca en el repo.

## Cloudflare

Account ID: entra a `https://dash.cloudflare.com/` y copia el ID en Workers and Pages Overview.
Tambien sale con `npx --yes wrangler@4 whoami`.
Guardalo como `CLOUDFLARE_ACCOUNT_ID`.

API token: crea uno en `https://dash.cloudflare.com/profile/api-tokens` con Create Custom Token.
Permisos minimos: Account Workers Scripts Edit, Queues Edit, R2 Edit, Pages Edit, D1 Edit para DO.
Guardalo como `CLOUDFLARE_API_TOKEN`.

Queue ID y bucket: lista con `npx --yes wrangler@4 queues list` y `npx --yes wrangler@4 r2 bucket list`.
Valores prod: `CLOUDFLARE_QUEUE_ID=vultus-jobs`, `R2_BUCKET=vultus-jobs`.
Preview usa `vultus-jobs-preview` y no necesita secreto extra.

R2 S3 keys: entra a R2 en `https://dash.cloudflare.com/` luego Manage R2 API Tokens y crea uno con Object Read and Write sobre `vultus-jobs`.
Copia Access Key ID como `R2_ACCESS_KEY_ID` y Secret como `R2_SECRET_ACCESS_KEY`.

API URL: fijo `VULTUS_API_URL=https://api.vultus.esau.com.mx`.
Frontend prod usa `VITE_API_URL` con el mismo valor en Pages.

## Modal

CLI: instala con `pip install 'modal==1.5.5'`.
Login: corre `modal token new` y sigue el link.
Te deja `MODAL_TOKEN_ID` y `MODAL_TOKEN_SECRET` en `~/.modal.toml`.
Copia esos dos a GitHub secrets.

Cloudflare dentro de Modal: crea el secreto una vez con `modal secret create vultus-cloudflare CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_API_TOKEN=... CLOUDFLARE_QUEUE_ID=vultus-jobs R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_BUCKET=vultus-jobs VULTUS_API_URL=https://api.vultus.esau.com.mx`.
Verifica con `modal secret list`.
Doc: `https://modal.com/docs/guide/secrets`.

Queues token extra: el consumer tambien lee `vultus-queues-token`.
Crealo con `modal secret create vultus-queues-token CLOUDFLARE_API_TOKEN=... CLOUDFLARE_ACCOUNT_ID=... CLOUDFLARE_QUEUE_ID=vultus-jobs`.
Si usas el mismo token de Cloudflare puedes repetir el valor.

Pesos: verifica con `modal volume list` que exista `vultus-weights`.
Doc: `https://modal.com/docs/guide/volumes`.

## GitHub

Crea el Environment en `https://github.com/dinoesau/vultus/settings/environments` con nombre `production`.
Agrega los 4 secrets: `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_ACCOUNT_ID`, `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`.
El workflow `.github/workflows/cd.yml` solo lee de ahi.
Verifica con un `workflow_dispatch` con `run_modal=false` primero.
