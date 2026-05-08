import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';

/* -------------------- GUARD -------------------- */

const HOME = os.homedir();
const PROD_DIR = path.join(HOME, 'kitsu');
const BETA_DIR = path.join(HOME, 'kitsu-beta');

const cwd = process.cwd();
if (cwd !== BETA_DIR) {
    console.error(`refusing: must run from ${BETA_DIR} (cwd is ${cwd})`);
    process.exit(1);
}

if (!fs.existsSync(PROD_DIR)) {
    console.error(`refusing: prod dir not found at ${PROD_DIR}`);
    process.exit(1);
}

const BETA_BACKEND_PORT = 8072;

/* -------------------- CONFIG -------------------- */

console.log(`[beta-sync] reading prod config: ${PROD_DIR}/config.json`);
const prodCfgRaw = fs.readFileSync(path.join(PROD_DIR, 'config.json'), 'utf-8');
const prodCfg = JSON.parse(prodCfgRaw);

if (!prodCfg.ports?.backend?.host || !prodCfg.ports?.backend?.port)
    throw new Error('prod config missing ports.backend.{host,port}');

const prodBackendURL = `http://${prodCfg.ports.backend.host}:${prodCfg.ports.backend.port}`;

const betaCfg = structuredClone(prodCfg);
betaCfg.ports.backend.port = BETA_BACKEND_PORT;
betaCfg.routedLlama = { url: prodBackendURL };

const betaCfgPath = path.join(BETA_DIR, 'config.json');
fs.writeFileSync(betaCfgPath, JSON.stringify(betaCfg, null, 4) + '\n');
console.log(`[beta-sync] wrote ${betaCfgPath}`);
console.log(`[beta-sync]   backend port: ${BETA_BACKEND_PORT}`);
console.log(`[beta-sync]   routedLlama.url: ${prodBackendURL}`);

/* -------------------- DATA DIR -------------------- */

const betaDataDir = path.join(BETA_DIR, 'backend', 'data');
fs.mkdirSync(betaDataDir, { recursive: true });

/* -------------------- DATABASE -------------------- */

const prodDb = path.join(PROD_DIR, 'backend', 'data', 'app.db');
const betaDb = path.join(betaDataDir, 'app.db');

if (!fs.existsSync(prodDb)) {
    console.error(`[beta-sync] prod db not found: ${prodDb} — skipping db sync`);
} else {
    console.log(`[beta-sync] backing up prod db: ${prodDb} -> ${betaDb}`);
    // sqlite3 .backup is the online-backup API: safe under concurrent writers.
    execFileSync('sqlite3', [prodDb, `.backup ${betaDb}`], { stdio: 'inherit' });
}

/* -------------------- UPLOADS -------------------- */

const prodUploads = path.join(PROD_DIR, 'backend', 'data', 'uploads') + '/';
const betaUploads = path.join(betaDataDir, 'uploads') + '/';

if (!fs.existsSync(prodUploads)) {
    console.log(`[beta-sync] prod uploads dir not found at ${prodUploads} — skipping`);
} else {
    fs.mkdirSync(betaUploads, { recursive: true });
    console.log(`[beta-sync] rsyncing uploads: ${prodUploads} -> ${betaUploads}`);
    execFileSync('rsync', ['-a', '--delete', prodUploads, betaUploads], { stdio: 'inherit' });
}

console.log(`[beta-sync] done`);
