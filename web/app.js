"use strict";

/* Great Lakes ice thickness, 1950-2025.

   Each layer is a pair of mipmapped uint8 textures: coverage and depth code.
   Coverage makes shorelines antialias and lets a lake smaller than a display
   pixel fade out instead of stippling the map. The fragment shader evaluates
   the v2 model per pixel:

       h = max(0, k * sqrt(max(0, S - C1 * ln d)) - m * P)

   A season is fetched, decoded and uploaded on demand, with the next two
   prefetched. Nothing is cached to storage. */

const FPS = 12;
const MAX_CACHE = 4;
const MAX_ZOOM = 10;

const LABELS = [
  { n: "Duluth", lon: -92.10, lat: 46.78, t: "city" },
  { n: "Thunder Bay", lon: -89.25, lat: 48.38, t: "city" },
  { n: "Marquette", lon: -87.40, lat: 46.54, t: "city" },
  { n: "Escanaba", lon: -87.06, lat: 45.75, t: "city" },
  { n: "Sault Ste. Marie", lon: -84.35, lat: 46.49, t: "city" },
  { n: "Green Bay", lon: -88.02, lat: 44.51, t: "city" },
  { n: "Milwaukee", lon: -87.91, lat: 43.04, t: "city" },
  { n: "Chicago", lon: -87.63, lat: 41.88, t: "city" },
  { n: "Traverse City", lon: -85.62, lat: 44.76, t: "city" },
  { n: "Alpena", lon: -83.43, lat: 45.06, t: "city" },
  { n: "Saginaw", lon: -83.95, lat: 43.42, t: "city" },
  { n: "Detroit", lon: -83.05, lat: 42.33, t: "city" },
  { n: "Cleveland", lon: -81.69, lat: 41.50, t: "city" },
  { n: "Sudbury", lon: -80.99, lat: 46.49, t: "city" },
  { n: "Sarnia", lon: -82.40, lat: 42.97, t: "city" },
  { n: "Owen Sound", lon: -80.94, lat: 44.57, t: "city" },
  { n: "Lake Superior", lon: -87.55, lat: 47.62, t: "lake" },
  { n: "Lake Michigan", lon: -86.85, lat: 44.15, t: "lake" },
  { n: "Lake Huron", lon: -82.35, lat: 44.85, t: "lake" },
  { n: "Georgian Bay", lon: -80.70, lat: 45.28, t: "lake" },
  { n: "Houghton Lake", lon: -84.76, lat: 44.33, t: "lake" },
  { n: "Torch Lake", lon: -85.32, lat: 45.02, t: "lake" },
  { n: "Burt Lake", lon: -84.66, lat: 45.47, t: "lake" },
  { n: "Glen Lake", lon: -86.05, lat: 44.90, t: "lake" }
];

const VERT = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2DArray;

uniform sampler2D u_glCov;
uniform sampler2D u_glDep;
uniform sampler2D u_inCov;
uniform sampler2D u_inDep;
uniform usampler2DArray u_S;
uniform usampler2DArray u_P;
uniform int u_frame;
uniform vec4 u_view;      // lonMin, latMin, lonMax, latMax
uniform vec4 u_glRect;    // left, top, lonSpan, latSpan
uniform vec4 u_inRect;
uniform vec4 u_field;     // lon0, lat0, step, unused
uniform vec2 u_fieldMax;
uniform vec3 u_model;     // k, C1, m
uniform float u_inlandFade; // 0 at the statewide view, 1 when zoomed in

in vec2 v_uv;
out vec4 outColor;

const vec3 LAND = vec3(0.047, 0.059, 0.078);
const vec3 OTHER = vec3(0.024, 0.055, 0.086);

float sat(int x, int y) { return float(texelFetch(u_S, ivec3(x, y, u_frame), 0).r); }
float pat(int x, int y) { return float(texelFetch(u_P, ivec3(x, y, u_frame), 0).r); }

void field(vec2 f, out float S, out float P) {
  f = clamp(f, vec2(0.0), u_fieldMax);
  ivec2 i0 = ivec2(floor(f));
  ivec2 i1 = min(i0 + ivec2(1), ivec2(u_fieldMax));
  vec2 t = f - vec2(i0);
  S = mix(mix(sat(i0.x, i0.y), sat(i1.x, i0.y), t.x),
          mix(sat(i0.x, i1.y), sat(i1.x, i1.y), t.x), t.y);
  P = mix(mix(pat(i0.x, i0.y), pat(i1.x, i0.y), t.x),
          mix(pat(i0.x, i1.y), pat(i1.x, i1.y), t.x), t.y);
}

vec3 ramp(float h) {
  float x = clamp(h / 20.32, 0.0, 1.0);
  // The floor is lifted off the land colour on purpose: in October, and in
  // any open-water lead, h = 0 must still read as water rather than as land.
  vec3 a = vec3(0.043, 0.098, 0.196);
  vec3 b = vec3(0.071, 0.208, 0.420);
  vec3 c = vec3(0.129, 0.369, 0.659);
  vec3 d = vec3(0.298, 0.620, 0.851);
  vec3 e = vec3(0.678, 0.863, 0.961);
  vec3 w = vec3(1.0);
  if (x < 0.18) return mix(a, b, x / 0.18);
  if (x < 0.42) return mix(b, c, (x - 0.18) / 0.24);
  if (x < 0.68) return mix(c, d, (x - 0.42) / 0.26);
  if (x < 0.86) return mix(d, e, (x - 0.68) / 0.18);
  return mix(e, w, (x - 0.86) / 0.14);
}

void layer(sampler2D covT, sampler2D depT, vec4 rect, float lon, float lat,
           out float cov, out float code) {
  vec2 uv = vec2((lon - rect.x) / rect.z, (rect.y - lat) / rect.w);
  if (uv.x < 0.0 || uv.x >= 1.0 || uv.y < 0.0 || uv.y >= 1.0) {
    cov = 0.0; code = 0.0; return;
  }
  cov = texture(covT, uv).r;
  float d = texture(depT, uv).r;
  code = cov > 0.002 ? d * 255.0 / cov : 0.0;
}

void main() {
  float lon = mix(u_view.x, u_view.z, v_uv.x);
  float lat = mix(u_view.y, u_view.w, v_uv.y);

  float cov, code, fade = 1.0;
  layer(u_glCov, u_glDep, u_glRect, lon, lat, cov, code);
  if (cov < 0.004) {
    layer(u_inCov, u_inDep, u_inRect, lon, lat, cov, code);
    fade = u_inlandFade;
  }
  cov *= fade;

  vec3 col = LAND;
  if (cov >= 0.004) {
    if (code >= 245.0) {
      col = mix(LAND, OTHER, cov);
    } else {
      float depth = exp(code / 40.0);
      float S, P;
      field(vec2((lon - u_field.x) / u_field.z,
                 (u_field.y - lat) / u_field.z), S, P);
      float arg = max(0.0, S - u_model.y * log(depth));
      float h = max(0.0, u_model.x * sqrt(arg) - u_model.z * P);
      col = mix(LAND, ramp(h), cov);
    }
  }
  outColor = vec4(col, 1.0);
}`;

/* Opening state. A season begins on 1 October, when the model has no ice
   anywhere, so opening there shows a dark, apparently empty map. Start on a
   mid-winter frame of the most recent season instead, which is also the most
   representative single view of the product. */
const OPEN_MONTH_DAY = "02-01";

const state = {
  gl: null, prog: null, vao: null, uni: {},
  tex: {}, layers: null, model: null,
  seasons: [], cur: null, curIndex: -1, frame: 0,
  playing: false, speed: 1, ready: false, switching: false,
  cache: new Map(), prefetch: new Map(),
  rect: { x: 0, y: 0, w: 1, h: 1 },
  base: null, view: null, zoom: 1, center: { lon: 0, lat: 0 },
  last: 0, acc: 0, dpr: 1
};

const $ = (id) => document.getElementById(id);

function fail(msg) {
  const e = $("error");
  e.hidden = false;
  e.textContent = msg;
  $("loading").classList.add("done");
  console.error(msg);
}

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) + "\n" + src);
  }
  return sh;
}

function buildProgram(gl) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  return p;
}

async function fetchJSON(url) {
  const r = await fetch(url, { cache: "no-cache" });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function gunzip(buf) {
  // Some hosts (or a proxy) transparently decode `Content-Encoding: gzip`, in
  // which case the body arrives already raw. Only inflate when the gzip magic
  // bytes are present, so the same files work behind any static host.
  const head = new Uint8Array(buf, 0, Math.min(2, buf.byteLength));
  if (!(head[0] === 0x1f && head[1] === 0x8b)) return buf;
  if (typeof DecompressionStream === "undefined") {
    throw new Error("This browser cannot decompress gzip (no DecompressionStream). Use a current Chrome, Edge, Firefox or Safari.");
  }
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
  return await new Response(stream).arrayBuffer();
}

function layerTexture(gl, bitmap) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, bitmap);
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

function fieldTexture(gl, w, h, n, data) {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D_ARRAY, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 2);
  gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R16UI, w, h, n, 0,
                gl.RED_INTEGER, gl.UNSIGNED_SHORT, data);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

async function loadSeason(i) {
  if (state.cache.has(i)) return state.cache.get(i);
  const meta = state.seasons[i];
  let buf = state.prefetch.get(i);
  if (!buf) {
    const r = await fetch("fields/" + meta.file, { cache: "force-cache" });
    if (!r.ok) throw new Error(`${meta.file}: HTTP ${r.status}`);
    buf = await r.arrayBuffer();
  }
  state.prefetch.delete(i);
  const raw = await gunzip(buf);
  const dv = new DataView(raw);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== "ICEF") throw new Error(`${meta.file}: bad magic`);
  const n = dv.getUint32(8, true);
  const h = dv.getUint32(12, true);
  const w = dv.getUint32(16, true);
  const bytes = n * h * w * 2;
  const gl = state.gl;
  const entry = {
    n, h, w,
    S: fieldTexture(gl, w, h, n, new Uint16Array(raw, 20, n * h * w)),
    P: fieldTexture(gl, w, h, n, new Uint16Array(raw, 20 + bytes, n * h * w))
  };
  state.cache.set(i, entry);
  while (state.cache.size > MAX_CACHE) {
    const oldest = state.cache.keys().next().value;
    if (oldest === i) break;
    const old = state.cache.get(oldest);
    gl.deleteTexture(old.S);
    gl.deleteTexture(old.P);
    state.cache.delete(oldest);
  }
  return entry;
}

function prefetch(i) {
  if (i < 0 || i >= state.seasons.length) return;
  if (state.prefetch.has(i) || state.cache.has(i)) return;
  fetch("fields/" + state.seasons[i].file)
    .then((r) => (r.ok ? r.arrayBuffer() : null))
    .then((b) => { if (b) state.prefetch.set(i, b); })
    .catch(() => {});
}

async function showSeason(i) {
  if (i < 0) i = 0;
  if (i >= state.seasons.length) i = 0;
  state.switching = true;
  try {
    const entry = await loadSeason(i);
    state.cur = entry;
    state.curIndex = i;
    state.frame = 0;
    state.ready = true;
    $("season").value = String(i);
    prefetch(i + 1);
    prefetch(i + 2);
  } finally {
    state.switching = false;
  }
}

function applyView() {
  const b = state.base, z = state.zoom;
  const halfLon = (b.lonMax - b.lonMin) / (2 * z);
  const halfLat = (b.latMax - b.latMin) / (2 * z);
  const lon = Math.min(Math.max(state.center.lon, b.lonMin + halfLon), b.lonMax - halfLon);
  const lat = Math.min(Math.max(state.center.lat, b.latMin + halfLat), b.latMax - halfLat);
  state.center = { lon, lat };
  state.view = {
    lonMin: lon - halfLon, lonMax: lon + halfLon,
    latMin: lat - halfLat, latMax: lat + halfLat
  };
}

function resize() {
  const canvas = $("map");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.dpr = dpr;
  const cw = Math.floor(canvas.clientWidth * dpr);
  const ch = Math.floor(canvas.clientHeight * dpr);
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  applyView();
  const v = state.view;
  const cos = Math.cos((v.latMin + v.latMax) * 0.5 * Math.PI / 180);
  const spanX = (v.lonMax - v.lonMin) * cos;
  const spanY = v.latMax - v.latMin;
  const scale = Math.min(cw / spanX, ch / spanY);
  const w = Math.round(spanX * scale);
  const h = Math.round(spanY * scale);
  state.rect = { x: Math.floor((cw - w) / 2), y: Math.floor((ch - h) / 2), w, h };
  placeLabels();
}

function placeLabels() {
  const box = $("labels");
  box.innerHTML = "";
  const v = state.view;
  if (!v) return;
  const cos = Math.cos((v.latMin + v.latMax) * 0.5 * Math.PI / 180);
  const spanX = (v.lonMax - v.lonMin) * cos;
  const spanY = v.latMax - v.latMin;
  const scale = Math.min(state.rect.w / spanX, state.rect.h / spanY);
  const padPx = 24;
  for (const l of LABELS) {
    if (l.lon < v.lonMin || l.lon > v.lonMax) continue;
    if (l.lat < v.latMin || l.lat > v.latMax) continue;
    const dx = (l.lon - (v.lonMin + v.lonMax) / 2) * cos * scale;
    const dy = ((v.latMin + v.latMax) / 2 - l.lat) * scale;
    const x = 50 + dx / state.rect.w * 100;
    const y = 50 + dy / state.rect.h * 100;
    if (x < -5 || x > 105 || y < -5 || y > 105) continue;
    if (Math.abs(dx) > state.rect.w / 2 - padPx || Math.abs(dy) > state.rect.h / 2 - padPx) continue;
    const el = document.createElement("div");
    el.className = "label " + l.t;
    el.textContent = l.n;
    el.style.left = x.toFixed(3) + "%";
    el.style.top = y.toFixed(3) + "%";
    box.appendChild(el);
  }
}

function screenToLL(clientX, clientY) {
  const canvas = $("map");
  const c = canvas.getBoundingClientRect();
  const px = (clientX - c.left) * state.dpr;
  const py = (clientY - c.top) * state.dpr;
  const r = state.rect, v = state.view;
  const fx = (px - r.x) / r.w;
  const fy = 1 - (py - r.y) / r.h;
  return {
    lon: v.lonMin + fx * (v.lonMax - v.lonMin),
    lat: v.latMin + fy * (v.latMax - v.latMin)
  };
}

function draw() {
  const gl = state.gl;
  if (!gl || !state.ready) return;
  const { x, y, w, h } = state.rect;
  gl.viewport(0, 0, gl.drawingBufferWidth, gl.drawingBufferHeight);
  gl.clearColor(0.02, 0.031, 0.051, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.viewport(x, y, w, h);

  gl.useProgram(state.prog);
  gl.bindVertexArray(state.vao);

  const bind = (unit, target, tex) => {
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(target, tex);
  };
  bind(0, gl.TEXTURE_2D, state.tex.glCov);
  bind(1, gl.TEXTURE_2D, state.tex.glDep);
  bind(2, gl.TEXTURE_2D, state.tex.inCov);
  bind(3, gl.TEXTURE_2D, state.tex.inDep);
  bind(4, gl.TEXTURE_2D_ARRAY, state.cur.S);
  bind(5, gl.TEXTURE_2D_ARRAY, state.cur.P);

  const u = state.uni;
  gl.uniform1i(u.u_glCov, 0);
  gl.uniform1i(u.u_glDep, 1);
  gl.uniform1i(u.u_inCov, 2);
  gl.uniform1i(u.u_inDep, 3);
  gl.uniform1i(u.u_S, 4);
  gl.uniform1i(u.u_P, 5);
  gl.uniform1i(u.u_frame, Math.min(state.frame, state.cur.n - 1));
  const v = state.view;
  gl.uniform4f(u.u_view, v.lonMin, v.latMin, v.lonMax, v.latMax);
  const g = state.layers.gl;
  gl.uniform4f(u.u_glRect, g.left, g.top, g.width * g.step, g.height * g.step);
  const n = state.layers.inland;
  gl.uniform4f(u.u_inRect, n.left, n.top, n.width * n.step, n.height * n.step);
  const f = state.field;
  gl.uniform4f(u.u_field, f.lon0, f.lat0, f.step, 0);
  gl.uniform2f(u.u_fieldMax, f.width - 1, f.height - 1);
  gl.uniform3f(u.u_model, state.model.k, state.model.C1, state.model.m);
  const t = Math.min(1, Math.max(0, (state.zoom - 1.8) / 1.4));
  gl.uniform1f(u.u_inlandFade, t * t * (3 - 2 * t));

  gl.drawArrays(gl.TRIANGLES, 0, 3);

  $("seasonLabel").textContent = state.seasons[state.curIndex].label;
  $("dateLabel").textContent = state.seasons[state.curIndex].frames[Math.min(state.frame, state.cur.n - 1)];
}

function tick(now) {
  const dt = Math.min(now - (state.last || now), 100);
  state.last = now;
  if (state.playing && state.ready && !state.switching) {
    state.acc += dt * state.speed;
    const per = 1000 / FPS;
    while (state.acc >= per) {
      state.acc -= per;
      state.frame += 1;
      if (state.frame >= state.cur.n) {
        const next = state.curIndex + 1 >= state.seasons.length ? 0 : state.curIndex + 1;
        state.frame = 0;
        showSeason(next);
        break;
      }
    }
  }
  draw();
  requestAnimationFrame(tick);
}

async function boot() {
  const canvas = $("map");
  const gl = canvas.getContext("webgl2", { antialias: false, alpha: false });
  if (!gl) throw new Error("This page needs WebGL2, which this browser did not provide.");
  state.gl = gl;
  const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE);
  if (maxTex < 8192) {
    throw new Error(`This GPU reports a maximum texture size of ${maxTex}px; the map needs 8192px.`);
  }

  const [layers, fields] = await Promise.all([
    fetchJSON("layers/manifest.json"),
    fetchJSON("fields/manifest.json")
  ]);
  state.layers = {
    gl: layers.layers.find((l) => l.name === "gl"),
    inland: layers.layers.find((l) => l.name === "inland")
  };
  const g = state.layers.gl;
  state.base = {
    lonMin: g.left,
    lonMax: g.left + g.width * g.step,
    latMax: g.top,
    latMin: g.top - g.height * g.step
  };
  state.center = {
    lon: (state.base.lonMin + state.base.lonMax) / 2,
    lat: (state.base.latMin + state.base.latMax) / 2
  };
  state.field = {
    lon0: fields.grid.lon[0],
    lat0: fields.grid.lat[0],
    step: fields.grid.lon[1] - fields.grid.lon[0],
    width: fields.grid.width,
    height: fields.grid.height
  };
  state.model = fields.model;
  state.seasons = fields.seasons;

  state.prog = buildProgram(gl);
  for (const name of ["u_glCov", "u_glDep", "u_inCov", "u_inDep", "u_S", "u_P",
                      "u_frame", "u_view", "u_glRect", "u_inRect", "u_field",
                      "u_fieldMax", "u_model", "u_inlandFade"]) {
    state.uni[name] = gl.getUniformLocation(state.prog, name);
  }

  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  state.vao = vao;

  const names = ["gl_cov", "gl_dep", "inland_cov", "inland_dep"];
  const bitmaps = await Promise.all(names.map(async (n) => {
    const r = await fetch(`layers/${n}.png`, { cache: "force-cache" });
    if (!r.ok) throw new Error(`layers/${n}.png: HTTP ${r.status}`);
    return createImageBitmap(await r.blob());
  }));
  state.tex.glCov = layerTexture(gl, bitmaps[0]);
  state.tex.glDep = layerTexture(gl, bitmaps[1]);
  state.tex.inCov = layerTexture(gl, bitmaps[2]);
  state.tex.inDep = layerTexture(gl, bitmaps[3]);

  resize();
  window.addEventListener("resize", resize);

  const slider = $("season");
  slider.max = String(state.seasons.length - 1);
  const qs = new URLSearchParams(location.search);
  // Default to the most recent season; `?season=1966` still selects one.
  const q = qs.get("season");
  const start = q === null
    ? state.seasons.length - 1
    : Math.max(0, Math.min(state.seasons.length - 1, Number(q) - 1950));
  await showSeason(Number.isFinite(start) ? start : state.seasons.length - 1);

  // Open on a mid-winter frame, not 1 October where there is no ice yet.
  const cur = state.seasons[state.curIndex];
  const target = `${Number(cur.season) + 1}-${OPEN_MONTH_DAY}`;
  const fi = cur.frames.findIndex((d) => d >= target);
  state.frame = fi < 0 ? 0 : fi;

  resize();
  $("loading").classList.add("done");
  requestAnimationFrame(tick);
}

function setPlaying(on) {
  state.playing = on;
  $("play").textContent = on ? "Pause" : "Play";
}

$("play").addEventListener("click", () => {
  if (!state.ready) return;
  setPlaying(!state.playing);
});
$("speed").addEventListener("change", (e) => { state.speed = Number(e.target.value); });
$("season").addEventListener("input", (e) => {
  const i = Number(e.target.value);
  if (i === state.curIndex) return;
  state.ready = false;
  showSeason(i).then(() => { state.ready = true; });
});

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  if (e.code === "ArrowRight" && state.ready) state.frame = Math.min(state.frame + 1, state.cur.n - 1);
  if (e.code === "ArrowLeft" && state.ready) state.frame = Math.max(state.frame - 1, 0);
});

/* ---------------------------------------------------------------- zoom/pan */
const canvas = $("map");
canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  if (!state.base) return;
  const before = screenToLL(e.clientX, e.clientY);
  const z2 = Math.min(MAX_ZOOM, Math.max(1, state.zoom * Math.exp(-e.deltaY * 0.0015)));
  if (Math.abs(z2 - state.zoom) < 1e-4) return;
  const b = state.base;
  const halfLon = (b.lonMax - b.lonMin) / (2 * z2);
  const halfLat = (b.latMax - b.latMin) / (2 * z2);
  const fx = (before.lon - state.view.lonMin) / (state.view.lonMax - state.view.lonMin);
  const fy = (before.lat - state.view.latMin) / (state.view.latMax - state.view.latMin);
  state.zoom = z2;
  state.center = {
    lon: before.lon - (fx - 0.5) * 2 * halfLon,
    lat: before.lat - (fy - 0.5) * 2 * halfLat
  };
  resize();
}, { passive: false });

let drag = null;
canvas.addEventListener("pointerdown", (e) => {
  drag = { x: e.clientX, y: e.clientY, lon: state.center.lon, lat: state.center.lat };
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = "grabbing";
});
canvas.addEventListener("pointermove", (e) => {
  if (!drag) return;
  const c = canvas.getBoundingClientRect();
  const cos = Math.cos((state.view.latMin + state.view.latMax) * 0.5 * Math.PI / 180);
  const dLon = -(e.clientX - drag.x) / c.width * (state.view.lonMax - state.view.lonMin);
  const dLat = (e.clientY - drag.y) / c.height * (state.view.latMax - state.view.latMin);
  state.center = { lon: drag.lon + dLon, lat: drag.lat + dLat };
  resize();
});
canvas.addEventListener("pointerup", () => {
  drag = null;
  canvas.style.cursor = "";
});
canvas.addEventListener("dblclick", () => {
  state.zoom = 1;
  state.center = {
    lon: (state.base.lonMin + state.base.lonMax) / 2,
    lat: (state.base.latMin + state.base.latMax) / 2
  };
  resize();
});

boot().catch((err) => fail(err && err.message ? err.message : String(err)));
