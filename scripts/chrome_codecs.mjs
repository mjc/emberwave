#!/usr/bin/env node
// Chrome's WebCodecs as a stand-in for NVENC (issue #82): encodes a Y4M with
// the codecs this Chrome has, decodes the streams again, and compares picture
// quality, size and time with emberwave's frames of the same size.
//
//   node scripts/chrome_codecs.mjs probe   [1080p|1440p|4k]
//   node scripts/chrome_codecs.mjs run     SIZE CODEC MBIT [options]
//   node scripts/chrome_codecs.mjs all     [options]
//   node scripts/chrome_codecs.mjs report  [DIR]    every run, one table a scene and size
//   node scripts/chrome_codecs.mjs summary [DIR]    the tables of BENCHMARKS.md
//
// Options:
//   run:  CODEC is a WebCodecs codec string (av01.0.08M.08, hvc1.1.6.L123.B0,
//         avc1.640028); --hw no-preference|prefer-hardware|prefer-software for
//         the encoder; --decode-hw for the decoder (default: as --hw; HEVC has
//         no software decoder in Chrome); --scene scene|grainy; --frames N;
//         --name LABEL
//   all:  --sizes 1080p,1440p,4k   --scenes scene,grainy   --codecs AV1,HEVC,H.264
//         --out DIR (default target/chrome-codecs)   --force (rerun what is
//         there)   --timeout MINUTES (a run's longest time, default 15)
//
// `all` writes one JSON file per run into DIR and, beside them, emberwave's
// results at the bytes each Chrome stream used; `summary` and `report` read
// them. It needs Google Chrome (CHROME names another binary than the macOS
// default), Node 20 or later (the script starts itself again with
// --experimental-websocket), the Y4M files of scripts/gen_bench_data.sh
// --sources and of scripts/chrome_codecs/make_grainy.py in testdata/generated,
// and a release build of emberwave-cli.
//
// Every run has a Chrome of its own: headless, with a mock keychain and a
// profile of its own, its page served on 127.0.0.1 only, so that nothing asks
// the user for a permission and a codec that hangs or crashes ends one run.
// Chrome, its server and its profile are removed at the end of the run, when
// the run times out, and on SIGINT and SIGTERM.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const CHROME = process.env.CHROME || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const CLI = join(root, 'target/release/emberwave-cli');

// The sizes of the other comparisons (BENCHMARKS.md): Moonlight's default bit
// rate at 60 frames a second, and emberwave's bytes per frame.
const SIZES = {
  '1080p': { moonlight: 20, emberwaveBytes: 400_000 },
  '1440p': { moonlight: 40, emberwaveBytes: 700_000 },
  '4k': { moonlight: 80, emberwaveBytes: 1_200_000 },
};
const DIMS = { '1080p': [1920, 1080], '1440p': [2560, 1440], '4k': [3840, 2160] };
// The pictures: the benchmark scene (scripts/gen_bench_data.sh --sources), and
// a copy of it with fresh noise in every frame (chrome_codecs/make_grainy.py).
const SCENES = ['scene', 'grainy'];
const FRAMERATE = 60;
const BOOLEAN_OPTIONS = new Set(['force']);

if (typeof WebSocket === 'undefined') {
  const again = spawnSync(process.execPath, ['--experimental-websocket', ...process.argv.slice(1)], { stdio: 'inherit' });
  process.exit(again.status ?? 1);
}

const dataDir = join(root, 'testdata/generated');
const sceneFile = (scene, size) => `${scene}_${size}_420.y4m`;
const sizeInfo = (name) => {
  const info = SIZES[name.toLowerCase()];
  if (!info) throw new Error(`unknown size ${name}: 1080p, 1440p or 4k`);
  return info;
};
const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

// What is running, so that a signal can end it.
const cleanups = new Set();
for (const signal of ['SIGINT', 'SIGTERM']) {
  process.on(signal, async () => {
    for (const cleanup of [...cleanups]) await cleanup();
    process.exit(130);
  });
}

// A static server on loopback: the page and the Y4M files, nothing else.
function serve() {
  const server = createServer((request, response) => {
    let file = null;
    try {
      const path = decodeURIComponent(new URL(request.url, 'http://x').pathname);
      if (path === '/' || path === '/page.html') {
        file = join(root, 'scripts/chrome_codecs/page.html');
      } else if (path.startsWith('/data/')) {
        const candidate = resolve(dataDir, path.slice(6));
        if (candidate.startsWith(dataDir + sep) && candidate.endsWith('.y4m')) file = candidate;
      }
    } catch {
      file = null;
    }
    if (!file || !existsSync(file)) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { 'content-length': statSync(file).size, 'content-type': file.endsWith('.html') ? 'text/html' : 'application/octet-stream' });
    createReadStream(file).pipe(response);
  });
  return new Promise((done) => server.listen(0, '127.0.0.1', () => done(server)));
}

// Chrome, headless, with a DevTools connection to a page. Everything it starts
// is ended by `close`, whatever fails after the start.
async function launch(url) {
  const profile = mkdtempSync(join(tmpdir(), 'emberwave-chrome-'));
  let chrome = null;
  let socket = null;
  let closing = null;
  const failPending = [];
  // Every call waits for the one that does the work: a failure that comes of
  // closing must not end the process before the profile is gone.
  const close = () => { closing ??= closeNow(); return closing; };
  const closeNow = async () => {
    cleanups.delete(close);
    for (const fail of failPending) fail(new Error('Chrome was closed'));
    try { socket?.close(); } catch { /* already closed */ }
    if (chrome && chrome.exitCode === null) {
      const ended = new Promise((done) => chrome.once('exit', done));
      // Chrome is a tree of processes: end the group it leads.
      try { process.kill(-chrome.pid, 'SIGTERM'); } catch { chrome.kill('SIGTERM'); }
      const killer = setTimeout(() => { try { process.kill(-chrome.pid, 'SIGKILL'); } catch { /* gone */ } }, 5000);
      await ended;
      clearTimeout(killer);
    }
    // Its helpers write to the profile until they have ended: try again.
    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        rmSync(profile, { recursive: true, force: true });
        return;
      } catch {
        await new Promise((r) => setTimeout(r, 300));
      }
    }
    console.error(`could not remove ${profile}`);
  };
  cleanups.add(close);
  try {
    chrome = spawn(CHROME, [
      '--headless=new', '--use-mock-keychain', '--password-store=basic', `--user-data-dir=${profile}`,
      '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
      '--autoplay-policy=no-user-gesture-required',
      // The page is never in front: do not let Chrome slow its timers or its tasks down.
      '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
      'about:blank',
    ], { stdio: ['ignore', 'ignore', 'pipe'], detached: true });
    const listening = await new Promise((done, fail) => {
      let text = '';
      const timer = setTimeout(() => fail(new Error('Chrome did not start in 30 s')), 30000);
      chrome.on('error', (error) => { clearTimeout(timer); fail(error); });
      chrome.on('exit', () => { clearTimeout(timer); fail(new Error('Chrome ended before it listened')); });
      chrome.stderr.on('data', (chunk) => {
        text += chunk;
        const match = /DevTools listening on (ws:\/\/\S+)/.exec(text);
        if (match) { clearTimeout(timer); done(match[1]); }
      });
    });
    const port = new URL(listening).port;
    const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${url}`, { method: 'PUT' })).json();
    socket = new WebSocket(target.webSocketDebuggerUrl);
    await new Promise((done, fail) => { socket.onopen = done; socket.onerror = fail; });
    let id = 0;
    const waiting = new Map();
    socket.onmessage = (message) => {
      const data = JSON.parse(message.data);
      if (data.id && waiting.has(data.id)) {
        waiting.get(data.id)(data);
        waiting.delete(data.id);
      }
    };
    socket.onclose = () => { for (const fail of failPending) fail(new Error('the DevTools connection closed')); };
    // Evaluates an expression and waits for its promise, for at most `ms`.
    const evaluate = (expression, ms) => new Promise((done, fail) => {
      failPending.push(fail);
      const timer = ms ? setTimeout(() => fail(new Error(`no answer in ${Math.round(ms / 60000)} minutes`)), ms) : null;
      waiting.set(++id, (reply) => {
        clearTimeout(timer);
        if (reply.error) fail(new Error(JSON.stringify(reply.error)));
        else if (reply.result.exceptionDetails) fail(new Error(reply.result.exceptionDetails.exception?.description || JSON.stringify(reply.result.exceptionDetails)));
        else done(reply.result.result.value);
      });
      socket.send(JSON.stringify({ id, method: 'Runtime.evaluate', params: { expression, awaitPromise: true, returnByValue: true } }));
    });
    return { evaluate, close };
  } catch (error) {
    await close();
    throw error;
  }
}

// A page of its own for `body`, on a server of its own.
async function withPage(body) {
  const server = await serve();
  let page = null;
  try {
    const base = `http://127.0.0.1:${server.address().port}`;
    page = await launch(`${base}/page.html`);
    for (let i = 0; i < 100 && !(await page.evaluate('window.ready === true', 5000)); i++) await new Promise((r) => setTimeout(r, 100));
    return await body(page, `${base}/data/`);
  } finally {
    await page?.close();
    server.close();
  }
}

// emberwave on the same pictures at a number of bytes per frame: the CLI
// encodes (on the GPU: the CPU's bytes are the same up to a few blocks) and
// `quality` compares each frame with the source. A budget the pictures' size
// does not allow is a result; a CLI that is missing or cannot run is not.
function emberwave(scene, size, bytesPerFrame, frames, work) {
  const input = join(dataDir, sceneFile(scene, size));
  const stream = join(work, `emberwave-${scene}-${size}-${bytesPerFrame}.pwv1`);
  const encode = spawnSync(CLI, ['encode', input, '--out', stream, '--bytes-per-frame', String(bytesPerFrame), '--frames', String(frames), '--gpu'], { encoding: 'utf8' });
  if (encode.status === null) throw new Error(`emberwave-cli did not run: ${encode.error?.message || encode.signal}`);
  if (encode.status !== 0) return { name: 'emberwave', scene, size, bytesPerFrame, error: `emberwave-cli encode: ${encode.stderr.trim().slice(0, 200)}` };
  const quality = spawnSync(CLI, ['quality', input, stream, '--frames', String(frames)], { encoding: 'utf8' });
  if (quality.status !== 0) throw new Error(`emberwave-cli quality: ${quality.stderr || quality.error?.message}`);
  const rows = [];
  for (const line of quality.stdout.split('\n')) {
    const m = /^frame\s+(\d+):\s+(\d+) bytes\s+Y\s+([\d.]+) dB\s+Cb\s+([\d.]+) dB\s+Cr\s+([\d.]+) dB/.exec(line);
    if (m) rows.push({ bytes: Number(m[2]), psnr: [Number(m[3]), Number(m[4]), Number(m[5])] });
  }
  if (rows.length !== frames) throw new Error(`emberwave-cli quality gave ${rows.length} frames, not ${frames}`);
  return { name: 'emberwave', scene, size, bytesPerFrame, rows };
}

function parseArgs(argv) {
  const positional = [];
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    if (!argv[i].startsWith('--')) positional.push(argv[i]);
    else if (BOOLEAN_OPTIONS.has(argv[i].slice(2))) options[argv[i].slice(2)] = true;
    else options[argv[i].slice(2)] = argv[++i];
  }
  return { positional, options };
}

// Why a run has no picture quality.
function failure(run) {
  if (run.error) return run.error;
  if (run.encoderError) return `encoder: ${run.encoderError}`;
  if (run.decoderError) return `its stream could not be decoded: ${run.decoderError}`;
  if (run.encoderSupported === false) return 'encoder not supported';
  if (run.decoderSupported === false) return 'decoder not supported';
  return 'no result';
}

const f = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? '' : v.toFixed(d));
const { positional, options } = parseArgs(process.argv.slice(2));
const command = positional[0];
const timeoutMs = Number(options.timeout || 15) * 60000;

if (command === 'probe') {
  const size = (positional[1] || '1080p').toLowerCase();
  const [width, height] = DIMS[size];
  const result = await withPage((page) => page.evaluate(`window.probe(${width}, ${height}, ${sizeInfo(size).moonlight * 1e6}, ${FRAMERATE})`, 60000));
  console.log(JSON.stringify(result, null, 2));
} else if (command === 'run') {
  const [size, codec, mbit] = positional.slice(1);
  if (!size || !codec || !mbit) throw new Error('run SIZE CODEC MBIT');
  const hw = options.hw || 'no-preference';
  const result = await withPage((page, base) => page.evaluate(`window.run(${JSON.stringify({
    url: base + sceneFile(options.scene || 'scene', size.toLowerCase()), codec, bitrate: Number(mbit) * 1e6, hw,
    decodeHw: options['decode-hw'] || hw, frames: Number(options.frames || 20), framerate: FRAMERATE, name: options.name || codec,
  })})`, timeoutMs));
  console.log(JSON.stringify(result));
} else if (command === 'all') {
  if (!existsSync(CLI)) throw new Error(`${CLI} is missing: cargo build --release -p emberwave-cli`);
  const out = resolve(options.out || join(root, 'target/chrome-codecs'));
  mkdirSync(out, { recursive: true });
  const sizes = (options.sizes || '1080p,1440p,4k').split(',').map((x) => x.toLowerCase());
  const scenes = (options.scenes || SCENES.join(',')).split(',');
  const frames = 20;
  const only = options.codecs ? options.codecs.split(',') : null;
  // A result is kept if it has pictures to compare; a run that failed is tried again.
  const done = (file) => !options.force && existsSync(file) && ('psnr' in JSON.parse(readFileSync(file, 'utf8')) || 'rows' in JSON.parse(readFileSync(file, 'utf8')));
  for (const size of sizes) {
    const info = sizeInfo(size);
    const [width, height] = DIMS[size];
    const probe = await withPage((page) => page.evaluate(`window.probe(${width}, ${height}, ${info.moonlight * 1e6}, ${FRAMERATE})`, 60000));
    writeFileSync(join(out, `probe-${size}.json`), JSON.stringify(probe, null, 1));
    const rates = [info.moonlight, Math.round(info.emberwaveBytes * 8 * FRAMERATE / 1e6)];
    for (const scene of scenes) {
      for (const [name, entry] of Object.entries(probe.codecs)) {
        // The codecs of NVENC: H.264, HEVC and AV1.
        if (name === 'VP9' || (only && !only.includes(name))) continue;
        // Encode in hardware where there is an encoder, and in software too;
        // decode where a client would, in hardware if there is a decoder.
        const decodeHw = entry.decode['prefer-hardware'] === true ? 'prefer-hardware' : 'prefer-software';
        const modes = ['prefer-hardware', 'prefer-software'].filter((hw) => entry.encode[hw] === true && entry.decode[decodeHw] === true);
        for (const hw of modes) {
          for (const mbit of rates) {
            const file = join(out, `${scene}-${size}-${name.replace(/[^A-Za-z0-9]/g, '')}-${hw.replace('prefer-', '')}-${mbit}.json`);
            if (done(file)) continue;
            console.error(`${scene} ${size} ${name} ${hw} ${mbit} Mbit/s`);
            let result;
            try {
              result = await withPage((page, base) => page.evaluate(`window.run(${JSON.stringify({ url: base + sceneFile(scene, size), codec: entry.codec, bitrate: mbit * 1e6, hw, decodeHw, frames, framerate: FRAMERATE, name })})`, timeoutMs));
            } catch (error) {
              result = { name, codec: entry.codec, hw, bitrate: mbit * 1e6, error: String(error.message || error) };
            }
            writeFileSync(file, JSON.stringify({ ...result, size, scene }));
          }
        }
      }
    }
  }
  // emberwave at the bytes each run actually used (all frames of the stream,
  // the key frame included), and at the budgets of the other comparisons.
  const budgets = new Map();
  for (const size of sizes) {
    const info = sizeInfo(size);
    for (const scene of scenes) {
      for (const bytes of [Math.round(info.moonlight * 1e6 / 8 / FRAMERATE), info.emberwaveBytes]) budgets.set(`${scene}-${size}-${bytes}`, { scene, size, bytes });
    }
  }
  for (const file of readdirSync(out)) {
    if (file.startsWith('probe') || file.startsWith('emberwave') || !file.endsWith('.json')) continue;
    const run = JSON.parse(readFileSync(join(out, file), 'utf8'));
    if (!run.chunkBytes || !sizes.includes(run.size) || !scenes.includes(run.scene)) continue;
    const bytes = Math.round(mean(run.chunkBytes) / 50) * 50;
    budgets.set(`${run.scene}-${run.size}-${bytes}`, { scene: run.scene, size: run.size, bytes });
  }
  for (const { scene, size, bytes } of budgets.values()) {
    const file = join(out, `emberwave-${scene}-${size}-${bytes}.json`);
    if (done(file)) continue;
    console.error(`emberwave ${scene} ${size} ${bytes} bytes a frame`);
    writeFileSync(file, JSON.stringify(emberwave(scene, size, bytes, frames, out)));
  }
} else if (command === 'report' || command === 'summary') {
  const dir = resolve(positional[1] || join(root, 'target/chrome-codecs'));
  const load = (file) => JSON.parse(readFileSync(join(dir, file), 'utf8'));
  const files = readdirSync(dir).filter((file) => file.endsWith('.json') && !file.startsWith('probe'));
  const chromeRuns = files.filter((file) => !file.startsWith('emberwave')).map(load);
  const steady = (rows) => rows.slice(1);
  const emberwaveAt = (run) => {
    const match = load(`emberwave-${run.scene}-${run.size}-${Math.round(mean(run.chunkBytes) / 50) * 50}.json`);
    return match.error ? null : match;
  };

  if (command === 'report') {
    // Every run, with emberwave at the bytes of each.
    const rows = [];
    for (const file of files.filter((x) => x.startsWith('emberwave'))) {
      const run = load(file);
      if (run.error) continue;
      rows.push({ scene: run.scene, size: run.size, who: 'emberwave', bytes: mean(run.rows.map((r) => r.bytes)), steadyBytes: mean(steady(run.rows).map((r) => r.bytes)), y: mean(steady(run.rows).map((r) => r.psnr[0])), yMin: Math.min(...steady(run.rows).map((r) => r.psnr[0])), cb: mean(steady(run.rows).map((r) => r.psnr[1])), cr: mean(steady(run.rows).map((r) => r.psnr[2])), key: run.rows[0].psnr[0] });
    }
    for (const run of chromeRuns) {
      const who = `${run.name} (${String(run.hw).replace('prefer-', '')} enc)`;
      if (!run.psnr) { rows.push({ scene: run.scene, size: run.size, who, asked: Math.round(run.bitrate / 1e6), error: failure(run), bytes: Infinity }); continue; }
      rows.push({ scene: run.scene, size: run.size, who, asked: Math.round(run.bitrate / 1e6), bytes: mean(run.chunkBytes), steadyBytes: mean(run.chunkBytes.slice(1)), y: mean(steady(run.psnr).map((p) => p[0])), yMin: Math.min(...steady(run.psnr).map((p) => p[0])), cb: mean(steady(run.psnr).map((p) => p[1])), cr: mean(steady(run.psnr).map((p) => p[2])), key: run.psnr[0][0], decode: median(run.decodeMs.slice(1)), queue: run.decodeQueueMsPerFrame, held: run.decoderHeld, encode: median(run.encodeMs.slice(1)) });
    }
    for (const scene of SCENES) {
      for (const size of Object.keys(SIZES)) {
        const part = rows.filter((r) => r.scene === scene && r.size === size).sort((a, b) => a.bytes - b.bytes || (a.asked ?? 0) - (b.asked ?? 0));
        if (!part.length) continue;
        console.log(`\n${scene}, ${size}\n`);
        console.log('| | asked Mbit/s | kB a frame (all) | kB after the key frame | Y dB | Cb dB | Cr dB | worst Y dB | key frame Y dB | decode ms | decode, queued ms | encode ms |');
        console.log('|---|---|---|---|---|---|---|---|---|---|---|---|');
        for (const r of part) {
          if (r.error) console.log(`| ${r.who} | ${r.asked ?? ''} | ${r.error} | | | | | | | | | |`);
          else console.log(`| ${r.who} | ${r.asked ?? ''} | ${f(r.bytes / 1000, 1)} | ${f(r.steadyBytes / 1000, 1)} | ${f(r.y)} | ${f(r.cb)} | ${f(r.cr)} | ${f(r.yMin)} | ${f(r.key)} | ${f(r.decode)}${r.held ? ' (held)' : ''} | ${f(r.queue)} | ${f(r.encode)} |`);
        }
      }
    }
  } else {
    // The encoders a host would use (the hardware ones, and AV1, which Chrome
    // encodes in software only), each next to emberwave at the same bytes a
    // frame, the key frame included.
    const order = { AV1: 0, HEVC: 1, 'H.264': 2 };
    const picked = chromeRuns.filter((r) => r.hw === 'prefer-hardware' || r.name === 'AV1');
    picked.sort((a, b) => SCENES.indexOf(a.scene) - SCENES.indexOf(b.scene) || Object.keys(SIZES).indexOf(a.size) - Object.keys(SIZES).indexOf(b.size) || a.bitrate - b.bitrate || order[a.name] - order[b.name]);
    const failures = [];
    for (const scene of SCENES) {
      console.log(`\n${scene}\n`);
      console.log('| Size | Codec | asked Mbit/s | kB a frame | kB after the key frame | codec Y dB | emberwave Y dB | codec Cb / Cr dB | emberwave Cb / Cr dB | decode ms | decode, queued ms |');
      console.log('|---|---|---|---|---|---|---|---|---|---|---|');
      for (const run of picked.filter((r) => r.scene === scene)) {
        const label = `${scene} ${run.size} ${run.name} ${Math.round(run.bitrate / 1e6)} Mbit/s`;
        if (!run.psnr) { failures.push(`${label}: ${failure(run)}`); continue; }
        const match = emberwaveAt(run);
        if (!match) { failures.push(`${label}: emberwave could not make frames of that size`); continue; }
        const e = (k) => mean(steady(match.rows).map((r) => r.psnr[k]));
        const c = (k) => mean(steady(run.psnr).map((p) => p[k]));
        console.log(`| ${run.size} | ${run.name} | ${Math.round(run.bitrate / 1e6)} | ${f(mean(run.chunkBytes) / 1000, 1)} | ${f(mean(run.chunkBytes.slice(1)) / 1000, 1)} | ${f(c(0))} | ${f(e(0))} | ${f(c(1))} / ${f(c(2))} | ${f(e(1))} / ${f(e(2))} | ${f(median(run.decodeMs.slice(1)))} | ${f(run.decodeQueueMsPerFrame)} |`);
      }
    }
    if (failures.length) console.log('\nNot measured:\n' + failures.map((x) => `- ${x}`).join('\n'));
  }
} else {
  console.error('usage: chrome_codecs.mjs probe [SIZE] | run SIZE CODEC MBIT [options] | all [options] | report [DIR] | summary [DIR]');
  process.exit(2);
}
