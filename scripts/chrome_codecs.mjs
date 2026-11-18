#!/usr/bin/env node
// Chrome's WebCodecs as a stand-in for NVENC (issue #82): encodes a Y4M with
// the codecs this Chrome has, decodes the streams again, and compares picture
// quality, size and time with emberwave's frames of the same size.
//
//   node scripts/chrome_codecs.mjs probe  [1080p|1440p|4k]
//   node scripts/chrome_codecs.mjs run    SIZE CODEC MBIT [--hw H] [--frames N]
//   node scripts/chrome_codecs.mjs all    [--sizes 1080p,1440p,4k] [--out DIR]
//   node scripts/chrome_codecs.mjs report [DIR]    every run
//   node scripts/chrome_codecs.mjs summary [DIR]   the tables of BENCHMARKS.md
//
// `all` writes one JSON file per run into DIR (target/chrome-codecs by
// default), `report` turns those into the tables of BENCHMARKS.md. It needs
// Google Chrome, Node 20 or later with --experimental-websocket (the script
// re-runs itself with the flag), the Y4M files of scripts/gen_bench_data.sh
// --sources in testdata/generated, and a release build of emberwave-cli.
//
// Chrome runs headless, with a mock keychain and a profile of its own, and its
// page is served on 127.0.0.1 only, so that nothing asks the user for a
// permission. Everything it starts is stopped at the end.

import { spawn, spawnSync } from 'node:child_process';
import { createServer } from 'node:http';
import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
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
// The pictures: the benchmark scene (scripts/gen_bench_data.sh --sources), and
// a copy of it with fresh noise in every frame (chrome_codecs/make_grainy.py).
const SCENES = ['scene', 'grainy'];
const sceneFile = (scene, size) => `${scene}_${size}_420.y4m`;
const FRAMERATE = 60;

if (typeof WebSocket === 'undefined') {
  const again = spawnSync(process.execPath, ['--experimental-websocket', ...process.argv.slice(1)], { stdio: 'inherit' });
  process.exit(again.status ?? 1);
}

// A static server on loopback: the page and the Y4M files.
function serve(directory) {
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://x').pathname);
    let file = null;
    if (path === '/' || path === '/page.html') file = join(root, 'scripts/chrome_codecs/page.html');
    else if (path.startsWith('/data/')) file = join(directory, path.slice(6));
    if (!file || file.includes('..') || !existsSync(file)) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, { 'content-length': statSync(file).size, 'content-type': file.endsWith('.html') ? 'text/html' : 'application/octet-stream' });
    createReadStream(file).pipe(response);
  });
  return new Promise((done) => server.listen(0, '127.0.0.1', () => done(server)));
}

// Chrome, headless, and a DevTools connection to a page.
async function launch(url) {
  const profile = mkdtempSync(join(tmpdir(), 'emberwave-chrome-'));
  const chrome = spawn(CHROME, [
    '--headless=new', '--use-mock-keychain', '--password-store=basic', `--user-data-dir=${profile}`,
    '--remote-debugging-port=0', '--no-first-run', '--no-default-browser-check', '--disable-extensions',
    '--autoplay-policy=no-user-gesture-required',
    // The page is never in front: do not let Chrome slow its timers or its tasks down.
    '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--disable-backgrounding-occluded-windows',
    'about:blank',
  ], { stdio: ['ignore', 'ignore', 'pipe'] });
  const version = await new Promise((done, fail) => {
    let text = '';
    chrome.stderr.on('data', (chunk) => {
      text += chunk;
      const match = /DevTools listening on (ws:\/\/\S+)/.exec(text);
      if (match) done(match[1]);
    });
    chrome.on('exit', () => fail(new Error('Chrome ended before it listened')));
    setTimeout(() => fail(new Error('Chrome did not start')), 30000);
  });
  const port = new URL(version).port;
  const target = await (await fetch(`http://127.0.0.1:${port}/json/new?${url}`, { method: 'PUT' })).json();
  const socket = new WebSocket(target.webSocketDebuggerUrl);
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
  const send = (method, params = {}) => new Promise((done) => {
    waiting.set(++id, done);
    socket.send(JSON.stringify({ id, method, params }));
  });
  const evaluate = async (expression) => {
    const reply = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true, timeout: 3_600_000 });
    if (reply.error) throw new Error(JSON.stringify(reply.error));
    if (reply.result.exceptionDetails) throw new Error(reply.result.exceptionDetails.exception?.description || JSON.stringify(reply.result.exceptionDetails));
    return reply.result.result.value;
  };
  // Chrome writes to its profile until it has ended: wait for it, then remove it.
  const close = async () => {
    try { socket.close(); } catch { /* already closed */ }
    const ended = new Promise((done) => chrome.once('exit', done));
    chrome.kill('SIGTERM');
    const killer = setTimeout(() => chrome.kill('SIGKILL'), 5000);
    await ended;
    clearTimeout(killer);
    rmSync(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
  };
  return { evaluate, close };
}

async function withPage(directory, body) {
  const server = await serve(directory);
  const page = await launch(`http://127.0.0.1:${server.address().port}/page.html`);
  try {
    for (let i = 0; i < 100 && !(await page.evaluate('window.ready === true')); i++) await new Promise((r) => setTimeout(r, 100));
    return await body(page, `http://127.0.0.1:${server.address().port}/data/`);
  } finally {
    await page.close();
    server.close();
  }
}

const dataDir = join(root, 'testdata/generated');
const sizeInfo = (name) => {
  const info = SIZES[name.toLowerCase()];
  if (!info) throw new Error(`unknown size ${name}: 1080p, 1440p or 4k`);
  return info;
};
const dims = { '1080p': [1920, 1080], '1440p': [2560, 1440], '4k': [3840, 2160] };

// emberwave on the same pictures at a number of bytes per frame: the CLI
// encodes (on the GPU: the CPU's bytes are the same up to a few blocks) and
// `quality` compares each frame with the source.
function emberwave(scene, size, bytesPerFrame, frames, work) {
  const input = join(dataDir, sceneFile(scene, size));
  const stream = join(work, `emberwave-${scene}-${size}-${bytesPerFrame}.pwv1`);
  const encode = spawnSync(CLI, ['encode', input, '--out', stream, '--bytes-per-frame', String(bytesPerFrame), '--frames', String(frames), '--gpu'], { encoding: 'utf8' });
  if (encode.status !== 0) throw new Error(`emberwave-cli encode: ${encode.stderr}`);
  const quality = spawnSync(CLI, ['quality', input, stream, '--frames', String(frames)], { encoding: 'utf8' });
  if (quality.status !== 0) throw new Error(`emberwave-cli quality: ${quality.stderr}`);
  const rows = [];
  for (const line of quality.stdout.split('\n')) {
    const m = /^frame\s+(\d+):\s+(\d+) bytes\s+Y\s+([\d.]+) dB\s+Cb\s+([\d.]+) dB\s+Cr\s+([\d.]+) dB/.exec(line);
    if (m) rows.push({ bytes: Number(m[2]), psnr: [Number(m[3]), Number(m[4]), Number(m[5])] });
  }
  return { name: 'emberwave', scene, size, bytesPerFrame, rows };
}

const mean = (values) => values.reduce((a, b) => a + b, 0) / values.length;
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

function parseArgs(argv) {
  const positional = [];
  const options = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i].startsWith('--')) options[argv[i].slice(2)] = argv[++i];
    else positional.push(argv[i]);
  }
  return { positional, options };
}

const { positional, options } = parseArgs(process.argv.slice(2));
const command = positional[0];

if (command === 'probe') {
  const size = (positional[1] || '1080p').toLowerCase();
  const [width, height] = dims[size];
  const result = await withPage(dataDir, (page) => page.evaluate(`window.probe(${width}, ${height}, ${sizeInfo(size).moonlight * 1e6}, ${FRAMERATE})`));
  console.log(JSON.stringify(result, null, 2));
} else if (command === 'run') {
  const [size, codec, mbit] = positional.slice(1);
  const hw = options.hw || 'no-preference';
  const frames = Number(options.frames || 20);
  const result = await withPage(dataDir, (page, base) => page.evaluate(`window.run(${JSON.stringify({
    url: base + sceneFile(options.scene || 'scene', size), codec, bitrate: Number(mbit) * 1e6, hw, decodeHw: options['decode-hw'] || hw, frames, framerate: FRAMERATE, name: options.name || codec,
  })})`));
  console.log(JSON.stringify(result));
} else if (command === 'all') {
  const out = resolve(options.out || join(root, 'target/chrome-codecs'));
  mkdirSync(out, { recursive: true });
  const sizes = (options.sizes || '1080p,1440p,4k').split(',');
  const scenes = (options.scenes || SCENES.join(',')).split(',');
  const frames = Number(options.frames || 20);
  const only = options.codecs ? options.codecs.split(',') : null;
  await withPage(dataDir, async (page, base) => {
    for (const size of sizes) {
      const info = sizeInfo(size);
      const [width, height] = dims[size];
      const probe = await page.evaluate(`window.probe(${width}, ${height}, ${info.moonlight * 1e6}, ${FRAMERATE})`);
      writeFileSync(join(out, `probe-${size}.json`), JSON.stringify(probe, null, 1));
      const rates = [info.moonlight, Math.round(info.emberwaveBytes * 8 * FRAMERATE / 1e6)];
      for (const scene of scenes) {
        for (const [name, entry] of Object.entries(probe.codecs)) {
          // The codecs of NVENC: H.264, HEVC and AV1.
          if (name === 'VP9' || (only && !only.includes(name))) continue;
          // Encode in hardware where there is an encoder and in software too; decode
          // where a client would, in hardware if there is a decoder.
          const decodeHw = entry.decode['prefer-hardware'] === true ? 'prefer-hardware' : 'prefer-software';
          const modes = ['prefer-hardware', 'prefer-software'].filter((hw) => entry.encode[hw] === true && (entry.decode[decodeHw] === true));
          for (const hw of modes) {
            for (const mbit of rates) {
              const file = join(out, `${scene}-${size}-${name.replace(/[^A-Za-z0-9]/g, '')}-${hw.replace('prefer-', '')}-${mbit}.json`);
              if (existsSync(file) && !options.force) continue;
              console.error(`${scene} ${size} ${name} ${hw} ${mbit} Mbit/s`);
              let result;
              try {
                result = await page.evaluate(`window.run(${JSON.stringify({ url: base + sceneFile(scene, size), codec: entry.codec, bitrate: mbit * 1e6, hw, decodeHw, frames, framerate: FRAMERATE, name })})`);
              } catch (error) {
                result = { name, codec: entry.codec, hw, bitrate: mbit * 1e6, error: String(error.message || error) };
              }
              result.size = size;
              result.scene = scene;
              writeFileSync(file, JSON.stringify(result));
            }
          }
        }
      }
    }
  });
  // emberwave at the bytes each run actually used (all frames of the stream,
  // the key frame included), where it can make a frame that small, and at the
  // budgets of the other comparisons.
  const budgets = new Map();
  for (const size of sizes) {
    const info = sizeInfo(size);
    for (const scene of scenes) {
      budgets.set(`${scene}-${size}-${Math.round(info.moonlight * 1e6 / 8 / FRAMERATE)}`, { scene, size, bytes: Math.round(info.moonlight * 1e6 / 8 / FRAMERATE) });
      budgets.set(`${scene}-${size}-${info.emberwaveBytes}`, { scene, size, bytes: info.emberwaveBytes });
    }
  }
  for (const file of readdirSync(out)) {
    if (file.startsWith('probe') || file.startsWith('emberwave') || !file.endsWith('.json')) continue;
    const run = JSON.parse(readFileSync(join(out, file), 'utf8'));
    if (!run.chunkBytes) continue;
    const bytes = Math.round(mean(run.chunkBytes) / 50) * 50;
    budgets.set(`${run.scene}-${run.size}-${bytes}`, { scene: run.scene, size: run.size, bytes });
  }
  for (const [, { scene, size, bytes }] of budgets) {
    const file = join(out, `emberwave-${scene}-${size}-${bytes}.json`);
    if (existsSync(file) && !options.force) continue;
    console.error(`emberwave ${scene} ${size} ${bytes} bytes a frame`);
    try {
      writeFileSync(file, JSON.stringify(emberwave(scene, size, bytes, frames, out)));
    } catch (error) {
      // A budget below the smallest frame the pictures' size allows.
      writeFileSync(file, JSON.stringify({ name: 'emberwave', scene, size, bytesPerFrame: bytes, error: String(error.message).slice(0, 200) }));
    }
  }
} else if (command === 'report') {
  const dir = resolve(positional[1] || join(root, 'target/chrome-codecs'));
  const runs = readdirSync(dir).filter((f) => f.endsWith('.json') && !f.startsWith('probe')).map((f) => JSON.parse(readFileSync(join(dir, f), 'utf8')));
  const rows = [];
  for (const run of runs) {
    if (run.name === 'emberwave') {
      if (run.error) continue;
      const all = run.rows;
      const steady = all.slice(1);
      rows.push({ scene: run.scene, size: run.size, who: 'emberwave', order: 1, asked: null, bytes: mean(all.map((r) => r.bytes)), steadyBytes: mean(steady.map((r) => r.bytes)), y: mean(steady.map((r) => r.psnr[0])), yMin: Math.min(...steady.map((r) => r.psnr[0])), cb: mean(steady.map((r) => r.psnr[1])), cr: mean(steady.map((r) => r.psnr[2])), key: all[0].psnr[0] });
    } else if (run.psnr) {
      const steady = run.psnr.slice(1);
      rows.push({ scene: run.scene, size: run.size, who: `${run.name} (${run.hw.replace('prefer-', '')} enc)`, order: 2, asked: Math.round(run.bitrate / 1e6), bytes: mean(run.chunkBytes), steadyBytes: mean(run.chunkBytes.slice(1)), keyBytes: run.chunkBytes[0], y: mean(steady.map((p) => p[0])), yMin: Math.min(...steady.map((p) => p[0])), cb: mean(steady.map((p) => p[1])), cr: mean(steady.map((p) => p[2])), key: run.psnr[0][0], decode: median(run.decodeMs.slice(1)), queue: run.decodeQueueMsPerFrame, held: run.decoderHeld, encode: median(run.encodeMs.slice(1)), keyframes: run.chunkTypes.filter((t) => t === 'key').length, decodeHw: run.decodeHw, sourceChange: mean(run.sourceChange) });
    } else {
      rows.push({ scene: run.scene, size: run.size, who: `${run.name} (${String(run.hw).replace('prefer-', '')} enc)`, order: 2, asked: Math.round(run.bitrate / 1e6), error: run.error || run.encoderError || run.decoderError || (run.encoderSupported === false ? 'encoder not supported' : run.decoderSupported === false ? 'decoder not supported' : 'no result') });
    }
  }
  const f = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? '' : v.toFixed(d));
  for (const scene of SCENES) {
    for (const size of Object.keys(SIZES)) {
      const part = rows.filter((r) => r.scene === scene && r.size === size);
      if (!part.length) continue;
      // Each codec's rows follow the emberwave row nearest to its bytes.
      part.sort((a, b) => a.bytes - b.bytes || a.order - b.order);
      console.log(`\n${scene}, ${size}\n`);
      console.log('| | asked Mbit/s | kB a frame (all) | kB after the key frame | Y dB | Cb dB | Cr dB | worst Y dB | key frame Y dB | decode ms | decode queue ms | encode ms |');
      console.log('|---|---|---|---|---|---|---|---|---|---|---|---|');
      for (const r of part) {
        if (r.error) console.log(`| ${r.who} | ${r.asked ?? ''} | ${r.error} | | | | | | | | | |`);
        else console.log(`| ${r.who} | ${r.asked ?? ''} | ${f(r.bytes / 1000, 1)} | ${f(r.steadyBytes / 1000, 1)} | ${f(r.y)} | ${f(r.cb)} | ${f(r.cr)} | ${f(r.yMin)} | ${f(r.key)} | ${f(r.decode)}${r.held ? ' (held)' : ''} | ${f(r.queue)} | ${f(r.encode)} |`);
      }
    }
  }
} else if (command === 'summary') {
  // The tables of BENCHMARKS.md: the encoders a host would use (the hardware
  // ones, and AV1, which Chrome encodes in software only), each next to
  // emberwave at the same bytes a frame.
  const dir = resolve(positional[1] || join(root, 'target/chrome-codecs'));
  const load = (f) => JSON.parse(readFileSync(join(dir, f), 'utf8'));
  const files = readdirSync(dir).filter((f) => f.endsWith('.json') && !f.startsWith('probe') && !f.startsWith('emberwave'));
  const order = { AV1: 0, HEVC: 1, 'H.264': 2 };
  const picked = files.map(load).filter((r) => r.hw === 'prefer-hardware' || r.name === 'AV1');
  picked.sort((a, b) => SCENES.indexOf(a.scene) - SCENES.indexOf(b.scene) || Object.keys(SIZES).indexOf(a.size) - Object.keys(SIZES).indexOf(b.size) || a.bitrate - b.bitrate || order[a.name] - order[b.name]);
  const f = (v, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? '' : v.toFixed(d));
  const failures = [];
  for (const scene of SCENES) {
    console.log(`\n${scene}\n`);
    console.log('| Size | Codec | asked Mbit/s | kB a frame | codec Y dB | emberwave Y dB | codec Cb / Cr dB | emberwave Cb / Cr dB | decode ms | decode, queued ms |');
    console.log('|---|---|---|---|---|---|---|---|---|---|');
    for (const run of picked.filter((r) => r.scene === scene)) {
      if (!run.psnr) {
        failures.push(`${scene} ${run.size} ${run.name} ${Math.round(run.bitrate / 1e6)} Mbit/s: ${run.error || run.encoderError || run.decoderError || 'no result'}`);
        continue;
      }
      const bytes = Math.round(mean(run.chunkBytes) / 50) * 50;
      const match = load(`emberwave-${scene}-${run.size}-${bytes}.json`);
      if (match.error) { failures.push(`${scene} ${run.size} ${run.name}: emberwave could not make frames of ${bytes} bytes`); continue; }
      const steady = (rows) => rows.slice(1);
      const ey = mean(steady(match.rows).map((r) => r.psnr[0])), ecb = mean(steady(match.rows).map((r) => r.psnr[1])), ecr = mean(steady(match.rows).map((r) => r.psnr[2]));
      const cy = mean(steady(run.psnr).map((p) => p[0])), ccb = mean(steady(run.psnr).map((p) => p[1])), ccr = mean(steady(run.psnr).map((p) => p[2]));
      console.log(`| ${run.size} | ${run.name} | ${Math.round(run.bitrate / 1e6)} | ${f(mean(run.chunkBytes) / 1000, 1)} | ${f(cy)} | ${f(ey)} | ${f(ccb)} / ${f(ccr)} | ${f(ecb)} / ${f(ecr)} | ${f(median(run.decodeMs.slice(1)))} | ${f(run.decodeQueueMsPerFrame)} |`);
    }
  }
  if (failures.length) console.log('\nNot measured:\n' + failures.map((x) => `- ${x}`).join('\n'));
} else {
  console.error('usage: chrome_codecs.mjs probe [SIZE] | run SIZE CODEC MBIT [--hw H] [--frames N] | all [--sizes S] [--codecs C] [--out DIR] | report [DIR] | summary [DIR]');
  process.exit(2);
}
