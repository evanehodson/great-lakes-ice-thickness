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
/* Phones have far less GPU memory than a desktop, and a tab that overshoots it
   loses its WebGL context, which leaves a blank canvas under working chrome.
   Coarse-pointer devices therefore get smaller layer textures and a shorter
   season cache. Raise MOBILE_TEX_CAP to 8192 if your phone copes. */
const COARSE = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
const MOBILE_TEX_CAP = 4096;
const MAX_CACHE = COARSE ? 3 : 4;
const MAX_ZOOM = 10;
/* Extra margin around the data extent, filled with the shader's land colour, so
   the lakes do not sit hard against the frame. */
const VIEW_PAD = 1.05;

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
layout(location = 0) in vec2 a_pos;
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

/* Sample unconditionally, with gradients taken before any clamping, and only
   then mask. Mipmapped lookups inside divergent branches have undefined
   derivatives, which tile-based mobile GPUs punish far more than desktop ones. */
void layer(sampler2D covT, sampler2D depT, vec4 rect, float lon, float lat,
           out float cov, out float code) {
  vec2 uv = vec2((lon - rect.x) / rect.z, (rect.y - lat) / rect.w);
  vec2 gx = dFdx(uv);
  vec2 gy = dFdy(uv);
  vec2 c = clamp(uv, vec2(0.0), vec2(1.0));
  float cv = textureGrad(covT, c, gx, gy).r;
  float dv = textureGrad(depT, c, gx, gy).r;
  bool inside = uv.x >= 0.0 && uv.x < 1.0 && uv.y >= 0.0 && uv.y < 1.0;
  cov = inside ? cv : 0.0;
  code = (inside && cv > 0.002) ? dv * 255.0 / cv : 0.0;
}

void main() {
  float lon = mix(u_view.x, u_view.z, v_uv.x);
  float lat = mix(u_view.y, u_view.w, v_uv.y);

  float cov, code, fade = 1.0;
  float cov2, code2;
  layer(u_glCov, u_glDep, u_glRect, lon, lat, cov, code);
  layer(u_inCov, u_inDep, u_inRect, lon, lat, cov2, code2);
  if (cov < 0.004) {
    cov = cov2;
    code = code2;
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
  pxW: 0, pxH: 0, onScreen: true,
  last: 0, acc: 0, dpr: 1, seq: 0
};

const $ = (id) => document.getElementById(id);

function deviceInfo() {
  const gl = state.gl;
  if (!gl) return "";
  const bits = [];
  try {
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    if (ext) bits.push(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL));
    bits.push("maxTex " + gl.getParameter(gl.MAX_TEXTURE_SIZE));
    bits.push("maxLayers " + gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS));
  } catch (_) { /* context may already be gone */ }
  bits.push("dpr " + (window.devicePixelRatio || 1));
  return "\n\n[" + bits.join(" | ") + "]";
}

function fail(msg) {
  const e = $("error");
  e.hidden = false;
  e.textContent = msg + deviceInfo();
  $("loading").classList.add("done");
  console.error(msg);
}

function progress(msg) {
  $("loading").textContent = msg;
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

/* The layer PNGs are single-channel 8-bit rasters. Handing them to
   createImageBitmap makes the browser expand each one to RGBA (256 MB for an
   8192 px square), colour-manage it, and decode all four at once, which is how a
   phone runs out of memory and loses its WebGL context. It can also silently
   alter the stored values, and the depth code lives in those values. So the
   PNG is decoded here to a raw byte array (64 MB for the same raster, one at a
   time), uploaded as R8, and box-halved first if the device cap is smaller. */

async function inflateZlib(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/* Returns { data, w, h } for an 8-bit, non-interlaced grey/RGB/RGBA PNG (first
   channel kept), or null when the file is anything else. */
async function decodePNG(buf) {
  if (typeof DecompressionStream === "undefined") return null;
  const u = new Uint8Array(buf);
  const dv = new DataView(buf);
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  for (let i = 0; i < 8; i++) if (u[i] !== sig[i]) return null;
  let p = 8, w = 0, h = 0, depth = 0, ctype = 0, interlace = 0, total = 0;
  const idat = [];
  while (p + 8 <= u.length) {
    const len = dv.getUint32(p);
    const type = String.fromCharCode(u[p + 4], u[p + 5], u[p + 6], u[p + 7]);
    const body = p + 8;
    if (type === "IHDR") {
      w = dv.getUint32(body); h = dv.getUint32(body + 4);
      depth = u[body + 8]; ctype = u[body + 9]; interlace = u[body + 12];
    } else if (type === "IDAT") {
      idat.push(u.subarray(body, body + len)); total += len;
    } else if (type === "IEND") break;
    p = body + len + 4;
  }
  if (!w || !h || depth !== 8 || interlace !== 0 || (ctype !== 0 && ctype !== 2 && ctype !== 6)) return null;
  const joined = new Uint8Array(total);
  let o = 0;
  for (const c of idat) { joined.set(c, o); o += c.length; }
  const raw = await inflateZlib(joined);
  const bpp = ctype === 0 ? 1 : ctype === 2 ? 3 : 4;
  const stride = w * bpp;
  if (raw.length < h * (stride + 1)) return null;
  const out = new Uint8Array(w * h);
  let prev = new Uint8Array(stride), cur = new Uint8Array(stride);
  let rp = 0;
  for (let y = 0; y < h; y++) {
    const ft = raw[rp++];
    cur.set(raw.subarray(rp, rp + stride));
    rp += stride;
    if (ft === 1) {
      for (let i = bpp; i < stride; i++) cur[i] += cur[i - bpp];
    } else if (ft === 2) {
      for (let i = 0; i < stride; i++) cur[i] += prev[i];
    } else if (ft === 3) {
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? cur[i - bpp] : 0;
        cur[i] += (a + prev[i]) >> 1;
      }
    } else if (ft === 4) {
      for (let i = 0; i < stride; i++) {
        const a = i >= bpp ? cur[i - bpp] : 0;
        const b = prev[i];
        const c = i >= bpp ? prev[i - bpp] : 0;
        const pp = a + b - c;
        const pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        cur[i] += (pa <= pb && pa <= pc) ? a : (pb <= pc ? b : c);
      }
    } else if (ft !== 0) return null;
    const row = y * w;
    if (bpp === 1) out.set(cur, row);
    else for (let x = 0; x < w; x++) out[row + x] = cur[x * bpp];
    const t = prev; prev = cur; cur = t;
  }
  return { data: out, w, h };
}

/* Box-average 2x2. Coverage and premultiplied depth code are both linear in the
   mean, so this matches what mipmapping would have produced. */
function halve(src, w, h) {
  const nw = (w + 1) >> 1, nh = (h + 1) >> 1;
  const dst = new Uint8Array(nw * nh);
  for (let y = 0; y < nh; y++) {
    const y0 = y * 2, y1 = Math.min(y0 + 1, h - 1);
    for (let x = 0; x < nw; x++) {
      const x0 = x * 2, x1 = Math.min(x0 + 1, w - 1);
      dst[y * nw + x] = (src[y0 * w + x0] + src[y0 * w + x1] +
                         src[y1 * w + x0] + src[y1 * w + x1] + 2) >> 2;
    }
  }
  return { data: dst, w: nw, h: nh };
}

function setLayerParams(gl) {
  gl.generateMipmap(gl.TEXTURE_2D);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}

/* Fallback for PNGs the decoder above does not handle, or browsers without
   DecompressionStream. Colour management and alpha handling are switched off,
   and an oversized raster is resized by the browser rather than uploaded whole. */
async function layerTextureFromBitmap(gl, buf, cap) {
  const blob = new Blob([buf], { type: "image/png" });
  const opts = { premultiplyAlpha: "none", colorSpaceConversion: "none" };
  let bmp = await createImageBitmap(blob, opts);
  if (Math.max(bmp.width, bmp.height) > cap) {
    const k = cap / Math.max(bmp.width, bmp.height);
    const rw = Math.max(1, Math.floor(bmp.width * k)), rh = Math.max(1, Math.floor(bmp.height * k));
    bmp.close();
    bmp = await createImageBitmap(blob, { ...opts, resizeWidth: rw, resizeHeight: rh, resizeQuality: "high" });
  }
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, bmp);
  bmp.close();
  setLayerParams(gl);
  return t;
}

async function layerTexture(gl, buf, cap) {
  let img = null;
  try { img = await decodePNG(buf); } catch (_) { img = null; }
  if (!img) return layerTextureFromBitmap(gl, buf, cap);
  while (Math.max(img.w, img.h) > cap) img = halve(img.data, img.w, img.h);
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, img.w, img.h, 0, gl.RED, gl.UNSIGNED_BYTE, img.data);
  setLayerParams(gl);
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
  const maxLayers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS);
  if (n > maxLayers) {
    throw new Error(`${meta.file}: ${n} frames, but this GPU allows only ${maxLayers} texture layers.`);
  }
  const entry = {
    n, h, w,
    S: fieldTexture(gl, w, h, n, new Uint16Array(raw, 20, n * h * w)),
    P: fieldTexture(gl, w, h, n, new Uint16Array(raw, 20 + bytes, n * h * w))
  };
  state.cache.set(i, entry);
  // Evict the oldest seasons, but never the one on screen or the one just loaded.
  for (const key of [...state.cache.keys()]) {
    if (state.cache.size <= MAX_CACHE) break;
    const old = state.cache.get(key);
    if (key === i || old === state.cur) continue;
    gl.deleteTexture(old.S);
    gl.deleteTexture(old.P);
    state.cache.delete(key);
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
  const token = ++state.seq;
  try {
    const entry = await loadSeason(i);
    if (token !== state.seq) return;   // a newer request has superseded this one
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
  if (!state.base || !state.pxW) return;
  const b = state.base;
  const { w, h, cos } = viewSize(state.zoom);
  const halfLon = (w / 2) / cos;
  const halfLat = h / 2;
  // At the fitted view the view is wider than the data, so the clamp has an
  // empty range; then the centre simply locks to the middle.
  state.center = {
    lon: clampAxis(state.center.lon, b.lonMin + halfLon, b.lonMax - halfLon),
    lat: clampAxis(state.center.lat, b.latMin + halfLat, b.latMax - halfLat)
  };
  state.view = {
    lonMin: state.center.lon - halfLon, lonMax: state.center.lon + halfLon,
    latMin: state.center.lat - halfLat, latMax: state.center.lat + halfLat
  };
}

function clampAxis(v, lo, hi) {
  return lo > hi ? (lo + hi) / 2 : Math.min(Math.max(v, lo), hi);
}

/* The view must match the canvas aspect, otherwise the map is letterboxed
   inside black bars and its edges cut the lakes. Size the view from the canvas,
   never from the raster extent, and keep a little margin where the shader
   paints land. */
function viewSize(zoom) {
  const b = state.base;
  const aspect = state.pxW / state.pxH;
  const cos = Math.cos(((b.latMin + b.latMax) * 0.5) * Math.PI / 180);
  const baseW = (b.lonMax - b.lonMin) * cos;
  const baseH = b.latMax - b.latMin;
  const h = Math.max(baseH, baseW / aspect) * VIEW_PAD / zoom;
  return { w: h * aspect, h, cos };
}

function resize() {
  const canvas = $("map");
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  state.dpr = dpr;
  const cw = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const ch = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  state.pxW = cw;
  state.pxH = ch;
  applyView();
  // The quad covers the whole canvas, so there is never a letterbox bar.
  state.rect = { x: 0, y: 0, w: cw, h: ch };
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
  const W = $("map").clientWidth;
  const H = $("map").clientHeight;
  const placed = [];
  // Lake names go first so that on a narrow screen they win over the cities
  // beside them; a label that would overlap one already placed, or run off the
  // edge, is simply left out until the reader zooms in.
  const ordered = LABELS.filter((l) => l.t === "lake")
    .concat(LABELS.filter((l) => l.t !== "lake"));
  for (const l of ordered) {
    if (l.lon < v.lonMin || l.lon > v.lonMax) continue;
    if (l.lat < v.latMin || l.lat > v.latMax) continue;
    const dx = (l.lon - (v.lonMin + v.lonMax) / 2) * cos * scale;
    const dy = ((v.latMin + v.latMax) / 2 - l.lat) * scale;
    const x = 50 + dx / state.rect.w * 100;
    const y = 50 + dy / state.rect.h * 100;
    if (x < -5 || x > 105 || y < -5 || y > 105) continue;

    const cx = x / 100 * W, cy = y / 100 * H;
    const bw = l.n.length * (l.t === "lake" ? 7.4 : 6.4) + 12;
    const bh = l.t === "lake" ? 20 : 24;
    const r = { l: cx - bw / 2, r: cx + bw / 2, t: cy - bh / 2, b: cy + bh / 2 };
    if (r.l < 4 || r.r > W - 4 || r.t < 4 || r.b > H - 4) continue;
    if (placed.some((p) => r.l < p.r + 4 && r.r > p.l - 4 && r.t < p.b + 2 && r.b > p.t - 2)) continue;
    placed.push(r);

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

function updateReadout() {
  if (state.curIndex < 0 || !state.seasons[state.curIndex]) return;
  $("seasonLabel").textContent = state.seasons[state.curIndex].label;
  $("dateLabel").textContent =
    state.seasons[state.curIndex].frames[Math.min(state.frame, state.cur.n - 1)];
}

/* The shader is per-pixel work, so redraw only when the frame, season, view or
   canvas size changed. Sitting paused on a phone then costs nothing. */
let lastKey = "";

function draw() {
  const gl = state.gl;
  if (!gl || !state.ready || !state.view) return;
  const vw = state.view;
  const key = [state.curIndex, state.frame, state.pxW, state.pxH,
               vw.lonMin, vw.latMin, vw.lonMax, vw.latMax].join("|");
  if (key === lastKey) return;
  lastKey = key;
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

  updateReadout();
}

function tick(now) {
  const dt = Math.min(now - (state.last || now), 100);
  state.last = now;
  if (!state.onScreen) {
    requestAnimationFrame(tick);
    return;
  }
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
  const texCap = Math.min(maxTex, COARSE ? MOBILE_TEX_CAP : 8192);
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    state.ready = false;
    fail("The browser dropped the GPU context, which usually means the page ran out of graphics memory. Reload the page; if it keeps happening, lower MOBILE_TEX_CAP in app.js.");
  });
  canvas.addEventListener("webglcontextrestored", () => location.reload());

  progress("Loading…");
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

  // Fetch the (small, compressed) files together, but decode and upload them one
  // at a time so only one raster is ever expanded in memory.
  const names = ["gl_cov", "gl_dep", "inland_cov", "inland_dep"];
  const keys = ["glCov", "glDep", "inCov", "inDep"];
  const files = await Promise.all(names.map(async (n) => {
    const r = await fetch(`layers/${n}.png`, { cache: "force-cache" });
    if (!r.ok) throw new Error(`layers/${n}.png: HTTP ${r.status}`);
    return r.arrayBuffer();
  }));
  for (let k = 0; k < names.length; k++) {
    progress(`Preparing map ${k + 1} / ${names.length}…`);
    state.tex[keys[k]] = await layerTexture(gl, files[k], texCap);
    files[k] = null;
  }
  progress("Loading season…");

  resize();
  // ResizeObserver rather than window "resize": it also fires on rotation and
  // when a phone's browser chrome changes the canvas box, and never reports a
  // stale size the way iOS Safari's resize event can.
  new ResizeObserver(() => resize()).observe(canvas);

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

  updateReadout();
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
  const tag = e.target && e.target.tagName;
  if (tag && /^(INPUT|SELECT|TEXTAREA|BUTTON|A)$/.test(tag)) return;
  if (e.code === "Space") { e.preventDefault(); setPlaying(!state.playing); }
  if (e.code === "ArrowRight" && state.ready) { state.frame = Math.min(state.frame + 1, state.cur.n - 1); updateReadout(); }
  if (e.code === "ArrowLeft" && state.ready) { state.frame = Math.max(state.frame - 1, 0); updateReadout(); }
});

/* ---------------------------------------------------------------- zoom/pan */
const canvas = $("map");

function zoomAt(factor, clientX, clientY) {
  const z2 = Math.min(MAX_ZOOM, Math.max(1, state.zoom * factor));
  if (Math.abs(z2 - state.zoom) < 1e-4) return;
  const before = screenToLL(clientX, clientY);
  const fx = (before.lon - state.view.lonMin) / (state.view.lonMax - state.view.lonMin);
  const fy = (before.lat - state.view.latMin) / (state.view.latMax - state.view.latMin);
  state.zoom = z2;
  const { w, h, cos } = viewSize(z2);
  state.center = {
    lon: before.lon - (fx - 0.5) * (w / cos),
    lat: before.lat - (fy - 0.5) * h
  };
  resize();
  syncTouchAction();
}

/* At the fitted view one finger scrolls the page; once zoomed in it pans the map. */
function syncTouchAction() {
  canvas.style.touchAction = state.zoom > 1.0001 ? "none" : "pan-y";
}

function resetView() {
  if (!state.base) return;
  state.zoom = 1;
  state.center = {
    lon: (state.base.lonMin + state.base.lonMax) / 2,
    lat: (state.base.latMin + state.base.latMax) / 2
  };
  resize();
  syncTouchAction();
}

canvas.addEventListener("wheel", (e) => {
  if (!state.base) return;
  // At the fitted view a plain wheel belongs to the page, so the notebook below
  // stays reachable. Zooming starts with ctrl/cmd-wheel, the +/- buttons, or
  // once the map is already zoomed in.
  if (state.zoom <= 1.0001 && !(e.ctrlKey || e.metaKey)) return;
  e.preventDefault();
  zoomAt(Math.exp(-e.deltaY * 0.0015), e.clientX, e.clientY);
}, { passive: false });

/* Pointer events cover mouse, touch and pen. One pointer pans; two pinch (zoom
   about the midpoint while the midpoint also pans). */
const pointers = new Map();   // pointerId -> { x, y } in client px
let pinch = null;             // { dist, mx, my } while two pointers are down

function panByPx(dx, dy) {
  if (!state.view) return;
  const c = canvas.getBoundingClientRect();
  const v = state.view;
  state.center = {
    lon: state.center.lon - dx / c.width * (v.lonMax - v.lonMin),
    lat: state.center.lat + dy / c.height * (v.latMax - v.latMin)
  };
  resize();
}

function pinchState() {
  const [a, b] = pointers.values();
  return {
    dist: Math.hypot(a.x - b.x, a.y - b.y),
    mx: (a.x + b.x) / 2,
    my: (a.y + b.y) / 2
  };
}

canvas.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  canvas.setPointerCapture(e.pointerId);
  canvas.style.cursor = "grabbing";
  if (pointers.size === 2) pinch = pinchState();
});

canvas.addEventListener("pointermove", (e) => {
  const p = pointers.get(e.pointerId);
  if (!p || !state.view) return;
  const dx = e.clientX - p.x;
  const dy = e.clientY - p.y;
  p.x = e.clientX;
  p.y = e.clientY;
  if (pointers.size >= 2 && pinch) {
    const now = pinchState();
    panByPx(now.mx - pinch.mx, now.my - pinch.my);
    if (pinch.dist > 0 && now.dist > 0) zoomAt(now.dist / pinch.dist, now.mx, now.my);
    pinch = now;
  } else if (pointers.size === 1) {
    panByPx(dx, dy);
  }
});

function endPointer(e) {
  pointers.delete(e.pointerId);
  pinch = pointers.size >= 2 ? pinchState() : null;
  if (pointers.size === 0) canvas.style.cursor = "";
}
// pointercancel fires when the browser takes over a vertical swipe to scroll the page.
canvas.addEventListener("pointerup", endPointer);
canvas.addEventListener("pointercancel", endPointer);
canvas.addEventListener("dblclick", resetView);

$("zoomIn").addEventListener("click", () => {
  const c = canvas.getBoundingClientRect();
  zoomAt(1.7, c.left + c.width / 2, c.top + c.height / 2);
});
$("zoomOut").addEventListener("click", () => {
  const c = canvas.getBoundingClientRect();
  zoomAt(1 / 1.7, c.left + c.width / 2, c.top + c.height / 2);
});
$("fit").addEventListener("click", resetView);

/* Stop drawing while the map has scrolled out of view; the animation keeps its
   position and resumes when the reader comes back. */
const stageObserver = new IntersectionObserver((entries) => {
  state.onScreen = entries[0].isIntersecting;
  lastKey = "";
}, { threshold: 0.02 });
stageObserver.observe($("stage"));
document.addEventListener("visibilitychange", () => { lastKey = ""; });

boot().catch((err) => fail(err && err.message ? err.message : String(err)));