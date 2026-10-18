"use strict";

(async function main() {
  const SCALE = 1e5;                       // probabilities are stored as integers in units of 1e-5
  const TARGETS = [
    { key: "label", name: "Any fire", long: "a wildfire start", keep: () => true },
    { key: "label_large", name: "Large (≥ 10 ha)", long: "a fire that reaches 10 ha", keep: (f) => f[4] >= 10 },
    { key: "label_lightning", name: "Lightning", long: "a lightning-caused fire", keep: (f) => f[3] === 0 },
    { key: "label_person", name: "Person-caused", long: "a person-caused fire", keep: (f) => f[3] === 1 },
  ];
  const CAUSE = ["Lightning", "Person", "Unknown cause"];
  const BINS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4];
  // one orange hue per mode: on light, low risk is pale and fades into the land; on dark, low
  // risk is a deep ember close to the background and high risk glows (not a flipped copy)
  const RAMPS = {
    light: { colors: ["#fff1e6", "#fdd0a8", "#f9a870", "#eb6834", "#c4491b", "#8f3010", "#5e1f0a"],
             opacity: [0.5, 0.58, 0.66, 0.74, 0.8, 0.86, 0.9] },
    dark: { colors: ["#4a2110", "#6b2b12", "#953a16", "#c4491b", "#e5642e", "#f68f55", "#fbbf8a"],
            opacity: [0.85, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98] },
  };
  const darkQuery = matchMedia("(prefers-color-scheme: dark)");
  const isDark = () => document.documentElement.dataset.theme === "dark"
    || (document.documentElement.dataset.theme !== "light" && darkQuery.matches);
  const ramp = () => RAMPS[isDark() ? "dark" : "light"];
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"];
  const DAY = 864e5;

  const $ = (id) => document.getElementById(id);
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
  const parse = (s) => new Date(`${s}T00:00:00Z`);
  const addDays = (d, n) => new Date(d.getTime() + n * DAY);
  const fmt = (d, opts) => d.toLocaleDateString("en-CA", { timeZone: "UTC", ...opts });
  const dShort = (d) => fmt(d, { day: "numeric", month: "short" });
  const dLong = (d) => fmt(d, { day: "numeric", month: "short", year: "numeric" });
  const pct = (p) => (p < 0.001 ? "<0.1%" : p < 0.1 ? `${(p * 100).toFixed(1)}%` : `${Math.round(p * 100)}%`);
  const count = (x) => (x < 10 ? x.toFixed(1) : Math.round(x).toString());
  const share = (x) => `${(x * 100).toFixed(1)}%`;

  // ------------------------------------------------------------------ data
  const get = (p) => fetch(p).then((r) => {
    if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
    return r.json();
  });
  let S, SEA, FT;
  try {
    [S, SEA, FT] = await Promise.all([get("data/static.json"), get("data/season.json"), get("data/fires_test.json")]);
  } catch (err) {
    $("week-title").textContent = "Could not load the data";
    $("week-sub").textContent = String(err);
    return;
  }
  const N = S.cells.length;
  const H = S.cellDeg / 2;

  // one global list of prediction dates: held-out test weeks, then this season's weeks
  const lastTest = S.weeks[S.weeks.length - 1];
  const weeks = [];
  S.weeks.forEach((date, i) => weeks.push({ date, d: parse(date), src: "test", i, live: false }));
  SEA.weeks.forEach((date, i) => {
    if (date > lastTest) weeks.push({ date, d: parse(date), src: "season", i, live: !SEA.complete[i] });
  });
  const G = weeks.length;
  const FIRES = {
    test: { origin: parse(FT.origin), list: FT.fires },
    season: { origin: parse(SEA.origin), list: SEA.fires },
  };
  const monthIdx = (g) => {
    const m = weeks[g].d.getUTCMonth() + 1;
    return Math.min(Math.max(m, S.months[0]), S.months[S.months.length - 1]) - S.months[0];
  };

  const cache = {};
  async function loadTarget(key) {
    if (cache[key]) return cache[key];
    const bt = await get(`data/backtest_${key}.json`);
    const risk = new Int32Array(G * N);
    const ev = new Uint8Array(G * N);
    const seasonToGlobal = new Map();
    weeks.forEach((w, g) => {
      if (w.src === "test") risk.set(bt.risk.slice(w.i * N, (w.i + 1) * N), g * N);
      else {
        risk.set(SEA.risk[key].slice(w.i * N, (w.i + 1) * N), g * N);
        seasonToGlobal.set(w.i, g);
      }
    });
    for (const [w, c] of bt.events) ev[w * N + c] = 1;
    for (const [w, c] of SEA.events[key]) if (seasonToGlobal.has(w)) ev[seasonToGlobal.get(w) * N + c] = 1;
    const clim = S.months.map((_, mi) => Int32Array.from(S.cells, (_c, c) => Math.max(S.clim[key][c][mi], 0)));
    const tot = weeks.map((w, g) => {
      let model = 0, normal = 0, actual = 0;
      const cl = clim[monthIdx(g)];
      for (let c = 0; c < N; c++) {
        const r = risk[g * N + c];
        if (r >= 0) model += r / SCALE;
        normal += cl[c] / SCALE;
        actual += ev[g * N + c];
      }
      return { model, normal, actual: w.live ? null : actual };
    });
    cache[key] = { key, risk, ev, clim, tot, orders: new Map() };
    return cache[key];
  }

  // cells ranked by risk within one week (stable on ties, missing values last)
  function ranking(td, g, src) {
    const id = src === "model" ? `m${g}` : `c${monthIdx(g)}`;
    if (!td.orders.has(id)) {
      const v = src === "model" ? td.risk.subarray(g * N, (g + 1) * N) : td.clim[monthIdx(g)];
      const idx = Array.from({ length: N }, (_, i) => i);
      idx.sort((a, b) => v[b] - v[a] || a - b);
      td.orders.set(id, idx);
    }
    return td.orders.get(id);
  }
  function covered(td, g, src, k) {
    const order = ranking(td, g, src);
    let hits = 0;
    for (let j = 0; j < k; j++) hits += td.ev[g * N + order[j]];
    return hits;
  }
  function pooled(td, k, which) {
    let m = 0, c = 0, tot = 0, n = 0;
    weeks.forEach((w, g) => {
      if (w.live || (which === "test") !== (w.src === "test")) return;
      m += covered(td, g, "model", k);
      c += covered(td, g, "clim", k);
      tot += td.tot[g].actual;
      n += 1;
    });
    return { m: m / tot, c: c / tot, n, tot };
  }
  function firesIn(g, keep) {
    const w = weeks[g];
    const set = FIRES[w.src];
    const t = Math.round((w.d - set.origin) / DAY);
    const list = set.list;
    let lo = 0, hi = list.length;           // first fire after day t (lists are sorted by day)
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid][2] <= t) lo = mid + 1; else hi = mid; }
    const out = [];
    for (let i = lo; i < list.length && list[i][2] <= t + 7; i++) if (keep(list[i])) out.push(list[i]);
    return out;
  }

  // ------------------------------------------------------------------ state
  const state = { t: "label", src: "model", w: G - 1, k: 30, cell: null };
  const initial = readHash();
  if (initial.t) state.t = initial.t;
  if (initial.g >= 0) state.w = initial.g;
  const target = () => TARGETS.find((t) => t.key === state.t);
  function readHash() {
    const h = new URLSearchParams(location.hash.slice(1));
    const t = TARGETS.some((x) => x.key === h.get("t")) ? h.get("t") : null;
    const g = h.get("w") ? weeks.findIndex((w) => w.date === h.get("w")) : -1;
    return { t, g };
  }

  // ------------------------------------------------------------------ map
  // No tile basemap: at 50 km cells street detail adds nothing, and drawing the province and a
  // few towns ourselves keeps the page free of third-party map keys and quotas.
  const TOWNS = [
    ["Vancouver", 49.28, -123.12], ["Victoria", 48.43, -123.37], ["Kelowna", 49.89, -119.5],
    ["Kamloops", 50.67, -120.33], ["Cranbrook", 49.51, -115.77], ["Revelstoke", 50.99, -118.2],
    ["Williams Lake", 52.14, -122.14], ["Prince George", 53.92, -122.75], ["Bella Coola", 52.37, -126.75],
    ["Port Hardy", 50.72, -127.5], ["Prince Rupert", 54.31, -130.32], ["Smithers", 54.78, -127.17],
    ["Fort St. John", 56.25, -120.85], ["Fort Nelson", 58.81, -122.7], ["Dease Lake", 58.44, -129.99],
    ["Atlin", 59.58, -133.69],
  ];
  const map = L.map("map", { zoomSnap: 0.25, minZoom: 4, maxZoom: 9, attributionControl: false });
  map.fitBounds([[48.3, -139.1], [60.0, -114.0]], { padding: [6, 6] });
  map.createPane("labels").style.zIndex = 450;
  map.getPane("labels").style.pointerEvents = "none";
  map.createPane("fires").style.zIndex = 460;
  const cellR = L.canvas({ padding: 0.3 });
  const fireR = L.canvas({ padding: 0.3, pane: "fires", tolerance: 5 });
  const land = L.geoJSON(S.boundary, { interactive: false, renderer: cellR }).addTo(map);
  const rects = S.cells.map(([, lat, lon], c) => L.rectangle([[lat - H, lon - H], [lat + H, lon + H]], {
    renderer: cellR, stroke: false, fillOpacity: 0,
  }).on("click", () => { state.cell = state.cell === c ? null : c; draw(); }).addTo(map));
  const boundary = L.geoJSON(S.boundary, { interactive: false, renderer: cellR }).addTo(map);
  const outlines = L.layerGroup().addTo(map);
  const fireLayer = L.layerGroup().addTo(map);
  for (const [name, lat, lon] of TOWNS) {
    L.marker([lat, lon], {
      pane: "labels", interactive: false, keyboard: false,
      icon: L.divIcon({ className: "town", html: `<i></i><span>${name}</span>`, iconSize: null }),
    }).addTo(map);
  }
  function styleBase() {
    land.setStyle({ stroke: false, fillColor: css("--land"), fillOpacity: 1 });
    boundary.setStyle({ color: css("--ink-2"), weight: 1, opacity: 0.6, fill: false });
  }
  styleBase();
  darkQuery.addEventListener("change", () => { styleBase(); drawLegend(); draw(); drawChart(); });

  const bin = (p) => { let i = 0; while (i < BINS.length && p >= BINS[i]) i++; return i; };   // 0 = below 0.5%

  function cellValues(td, g) {
    return state.src === "model" ? td.risk.subarray(g * N, (g + 1) * N) : td.clim[monthIdx(g)];
  }

  function firePopup(f, src) {
    const set = FIRES[src];
    const date = addDays(set.origin, f[2]);
    let html = `<b>${esc(dLong(date))}</b><br>${CAUSE[f[3]]} · ${f[4] >= 1 ? Math.round(f[4]).toLocaleString() : f[4]} ha`;
    if (src === "season" && typeof f[5] === "string" && /^[A-Z0-9]+$/.test(f[5])) {
      const url = `https://wildfiresituation.nrs.gov.bc.ca/incidents?fireYear=${SEA.year}&incidentNumber=${f[5]}`;
      html += `<br><a href="${url}" target="_blank" rel="noopener">${esc(f[6] || f[5])}</a>`;
    }
    return html;
  }

  // ------------------------------------------------------------------ panel + map update
  function draw() {
    const td = cache[state.t];
    if (!td) return;
    const g = state.w;
    const w = weeks[g];
    const tgt = target();
    const vals = cellValues(td, g);
    const ink = css("--ink"), surface = css("--surface"), accent = css("--model");
    const { colors, opacity } = ramp();

    for (let c = 0; c < N; c++) {
      const v = vals[c];
      const b = v < 0 ? 0 : bin(v / SCALE);
      rects[c].setStyle(b === 0 ? { fillOpacity: 0 } : { fillColor: colors[b - 1], fillOpacity: opacity[b - 1] });
    }
    outlines.clearLayers();
    const order = ranking(td, g, state.src);
    for (let j = 0; j < state.k; j++) {
      const [, lat, lon] = S.cells[order[j]];
      L.rectangle([[lat - H, lon - H], [lat + H, lon + H]], {
        renderer: cellR, interactive: false, color: ink, weight: 1.25, opacity: 0.9, fill: false,
      }).addTo(outlines);
    }
    if (state.cell !== null) {
      const [, lat, lon] = S.cells[state.cell];
      L.rectangle([[lat - H, lon - H], [lat + H, lon + H]], {
        renderer: cellR, interactive: false, color: accent, weight: 3, fill: false,
      }).addTo(outlines);
    }
    fireLayer.clearLayers();
    const fires = firesIn(g, tgt.keep);
    for (const f of fires) {
      L.circleMarker([f[0], f[1]], {
        renderer: fireR, radius: 3.5, weight: 1.5, color: surface, fillColor: ink, fillOpacity: 1,
      }).bindPopup(() => firePopup(f, w.src)).addTo(fireLayer);
    }

    // header
    const badge = $("badge");
    badge.className = `badge ${w.live ? "live" : w.src === "season" ? "season" : ""}`;
    badge.textContent = w.live ? "Live nowcast" : w.src === "season" ? `${SEA.year} · forward test` : "Held-out test year";
    const start = addDays(w.d, 1), end = addDays(w.d, 7);
    $("week-title").textContent = `${dShort(start)} – ${dLong(end)}`;
    let sub;
    if (w.live) {
      sub = `Nowcast from weather through ${dShort(w.d)} (NASA data runs about 3 days behind).`;
      if (!SEA.inSeason) sub += " This is outside the April–October season the model covers.";
      const age = (Date.now() - parse(SEA.generated.slice(0, 10))) / DAY;
      if (age > 9) sub += ` Last refreshed ${dLong(parse(SEA.generated.slice(0, 10)))}; it updates weekly in season.`;
    } else if (w.src === "season") {
      sub = `Scored using only data up to ${dShort(w.d)}. The model was frozen before this season began.`;
    } else {
      sub = `Predicted from data up to ${dShort(w.d)}. The model never saw ${w.d.getUTCFullYear()}: it was trained on 2005–2018.`;
    }
    $("week-sub").textContent = sub;
    $("prev").disabled = g === 0;
    $("next").disabled = g === G - 1;
    $("latest").disabled = g === G - 1;

    // tiles
    const tot = td.tot[g];
    $("t-model").textContent = count(tot.model);
    $("t-clim").textContent = count(tot.normal);
    $("t-month").textContent = MONTHS[S.months[monthIdx(g)] - 1];
    if (w.live) {
      $("t-actual-k").textContent = "Reported so far";
      $("t-actual").textContent = fires.length;
      $("t-actual-u").textContent = fires.length === 1 ? "fire (window still open)" : "fires (window still open)";
    } else {
      $("t-actual-k").textContent = "Actually had one";
      $("t-actual").textContent = tot.actual;
      $("t-actual-u").textContent = `cells · ${fires.length} fire${fires.length === 1 ? "" : "s"}`;
    }

    // crew game
    $("k-out").textContent = state.k;
    const k = state.k;
    if (w.live) {
      $("crew-result").innerHTML = `<p class="fine" style="margin:0">The outlined cells on the map are the ${k} the ${state.src === "model" ? "model" : "historical average"} would watch this week. They can be scored once the week is over.</p>`;
    } else if (tot.actual === 0) {
      $("crew-result").innerHTML = `<p class="fine" style="margin:0">No cell had ${esc(tgt.long)} this week, so there is nothing to cover.</p>`;
    } else {
      const rows = [
        ["Model's top", covered(td, g, "model", k), "model"],
        ["Normal's top", covered(td, g, "clim", k), "clim"],
        ["Random pick", (tot.actual * k) / N, "rand"],
      ];
      $("crew-result").innerHTML = `<div class="cov">${rows.map(([name, hits, cls]) => `
        <span>${name} ${cls === "rand" ? "" : k}</span>
        <span class="bar"><i class="${cls}" style="width:${(100 * hits) / tot.actual}%"></i></span>
        <span class="n">${cls === "rand" ? "~" + hits.toFixed(1) : hits} of ${tot.actual}</span>`).join("")}</div>`;
    }
    const all = pooled(td, k, "test");
    const sea = pooled(td, k, "season");
    let txt = `Over all ${all.n} test weeks, watching ${k} cells a week covers ${share(all.m)} of cells with a fire using the model, ${share(all.c)} using the normal and ${share(k / N)} at random.`;
    if (sea.n && sea.tot) txt += ` ${SEA.year} so far (${sea.n} weeks): model ${share(sea.m)}, normal ${share(sea.c)}.`;
    $("crew-all").textContent = txt;

    drawCell(td, g, fires);
    const h = new URLSearchParams({ w: w.date, t: state.t });
    history.replaceState(null, "", `#${h}`);
  }

  function drawCell(td, g, fires) {
    const box = $("cell-info");
    if (state.cell === null) {
      box.innerHTML = '<p class="fine" style="margin:0">Click a cell on the map for its numbers.</p>';
      return;
    }
    const c = state.cell;
    const w = weeks[g];
    const [, lat, lon] = S.cells[c];
    const r = td.risk[g * N + c];
    const n = td.clim[monthIdx(g)][c] / SCALE;
    const inCell = fires.filter((f) => Math.abs(f[0] - lat) <= H && Math.abs(f[1] - lon) <= H).length;
    const rank = ranking(td, g, "model").indexOf(c) + 1;
    const rows = [
      ["This week (model)", r < 0 ? "no weather data" : `<span class="big">${pct(r / SCALE)}</span>`],
      [`Normal for ${MONTHS[S.months[monthIdx(g)] - 1]}`, pct(n)],
    ];
    if (r >= 0 && n > 0) rows.push(["Compared with normal", `${(r / SCALE / n).toFixed(1)}×`]);
    rows.push(["Model rank this week", `${rank} of ${N}`]);
    rows.push([w.live ? "Fires so far" : "Fires in the window", String(inCell)]);
    if (w.live) {
      const nw = SEA.now;
      const v = (key, unit, dp = 0) => (nw[key][c] === null ? "–" : `${nw[key][c].toFixed(dp)}${unit}`);
      rows.push(["Fire Weather Index", v("fwi", "", 1)], ["Drought Code", v("dc", "")],
        ["Max temperature", v("tmax", " °C", 1)], ["Humidity (at max temp)", v("rh", "%")],
        ["Max wind", v("wind_kmh", " km/h")], ["Rain, last 30 days", v("precip_30d", " mm")],
        ["Days since rain", v("days_since_rain", "")]);
    }
    box.innerHTML = `<h3>Cell at ${lat.toFixed(2)}° N, ${Math.abs(lon).toFixed(2)}° W</h3>
      <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;
  }

  function drawLegend() {
    const ticks = ["0.5", "1", "2", "5", "10", "20", "40%"];
    const { colors, opacity } = ramp();
    $("legend").innerHTML = `<div>Chance of ${esc(target().long)} in the 7 days</div>
      <div class="ramp">${colors.map((c, i) => `<i style="background:${c};opacity:${opacity[i]}"></i>`).join("")}</div>
      <div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join("")}</div>
      <div class="keys"><span><i class="dot"></i>Fire that started in the window</span>
      <span><i class="box"></i>Cells crews would watch</span><span>Blank: under 0.5%</span></div>`;
  }

  // ------------------------------------------------------------------ timeline
  const chart = $("chart");
  const tip = $("tip");
  const years = [...new Set(weeks.map((w) => w.d.getUTCFullYear()))];
  const GAP = 3, SPAN = 31;
  const posOf = weeks.map((w) => {
    const y = w.d.getUTCFullYear();
    return years.indexOf(y) * (SPAN + GAP) + (w.d - Date.UTC(y, 3, 1)) / (7 * DAY);
  });
  const xMax = years.length * (SPAN + GAP) - GAP;
  let geom = null;

  function drawChart() {
    const td = cache[state.t];
    if (!td) return;
    const W = chart.clientWidth - 8, Ht = chart.clientHeight - 8;
    const m = { l: 36, r: 12, t: 22, b: 22 };
    const iw = W - m.l - m.r, ih = Ht - m.t - m.b;
    const x = (g) => m.l + (posOf[g] / xMax) * iw;
    let top = 0;
    td.tot.forEach((t) => { top = Math.max(top, t.model, t.normal, t.actual ?? 0); });
    const step = [1, 2, 5, 10, 20, 25, 50, 100].find((s) => top / s <= 4) || 200;
    const yMax = Math.max(step, Math.ceil(top / step) * step);
    const y = (v) => m.t + ih - (v / yMax) * ih;
    const unit = iw / xMax;
    const bw = Math.max(1, Math.min(24, unit * 0.62));
    let s = "";
    // forward-test band
    const seasonG = weeks.map((w, g) => (w.src === "season" ? g : -1)).filter((g) => g >= 0);
    if (seasonG.length) {
      const yr = weeks[seasonG[0]].d.getUTCFullYear();
      const x0 = m.l + ((years.indexOf(yr) * (SPAN + GAP) - 0.8) / xMax) * iw;
      const x1 = m.l + ((years.indexOf(yr) * (SPAN + GAP) + SPAN) / xMax) * iw;
      const xr = Math.min(x1, W - m.r);
      s += `<rect class="band" x="${x0}" y="${m.t - 16}" width="${xr - x0}" height="${ih + 16}" rx="4"/>`;
      s += xr - x0 > 130   // on narrow screens the label ends at the band's edge and runs left
        ? `<text class="band-label" x="${x0 + 6}" y="${m.t - 4}">${yr} · forward test</text>`
        : `<text class="band-label" x="${xr - 4}" y="${m.t - 4}" text-anchor="end">${yr} · forward test</text>`;
    }
    for (let v = 0; v <= yMax; v += step) {
      s += `<line class="${v === 0 ? "baseline" : "gridline"}" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>`;
      s += `<text class="tick" x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`;
    }
    years.forEach((yr, i) => {
      const xc = m.l + ((i * (SPAN + GAP) + SPAN / 2) / xMax) * iw;
      s += `<text class="year" x="${xc}" y="${Ht - 4}" text-anchor="middle">${yr}</text>`;
    });
    td.tot.forEach((t, g) => {
      if (t.actual) s += `<rect class="bar" x="${x(g) - bw / 2}" y="${y(t.actual)}" width="${bw}" height="${y(0) - y(t.actual)}" rx="${bw > 3 ? 1 : 0}"/>`;
    });
    for (const yr of years) {
      const gs = weeks.map((w, g) => (w.d.getUTCFullYear() === yr ? g : -1)).filter((g) => g >= 0);
      const path = (key) => gs.map((g, j) => `${j ? "L" : "M"}${x(g).toFixed(1)},${y(td.tot[g][key]).toFixed(1)}`).join("");
      s += `<path class="l-clim" d="${path("normal")}"/><path class="l-model" d="${path("model")}"/>`;
    }
    const live = weeks.findIndex((w) => w.live);
    if (live >= 0) s += `<circle class="live-dot" cx="${x(live)}" cy="${y(td.tot[live].model)}" r="4.5"/>`;
    s += `<line class="sel" x1="${x(state.w)}" x2="${x(state.w)}" y1="${m.t - 16}" y2="${y(0)}"/>`;
    s += `<line class="hover" id="hover-line" x1="0" x2="0" y1="${m.t - 16}" y2="${y(0)}" visibility="hidden"/>`;
    s += `<rect x="${m.l}" y="0" width="${iw}" height="${Ht}" fill="transparent" id="hit"/>`;
    chart.querySelector("svg")?.remove();
    chart.insertAdjacentHTML("afterbegin", `<svg viewBox="0 0 ${W} ${Ht}" role="img" aria-label="Weekly expected and actual cells with a fire, ${years[0]} to ${years[years.length - 1]}">${s}</svg>`);
    geom = { x, W };
  }

  function nearest(evt) {
    const svg = chart.querySelector("svg");
    const box = svg.getBoundingClientRect();
    const px = ((evt.clientX - box.left) / box.width) * geom.W;
    let best = 0, bd = Infinity;
    for (let g = 0; g < G; g++) { const d = Math.abs(geom.x(g) - px); if (d < bd) { bd = d; best = g; } }
    return { g: best, px: (geom.x(best) / geom.W) * box.width };
  }
  chart.addEventListener("pointermove", (evt) => {
    if (!geom || !cache[state.t]) return;
    const { g, px } = nearest(evt);
    const t = cache[state.t].tot[g];
    const w = weeks[g];
    const line = chart.querySelector("#hover-line");
    line.setAttribute("x1", geom.x(g)); line.setAttribute("x2", geom.x(g)); line.setAttribute("visibility", "visible");
    tip.innerHTML = `<b>${dShort(addDays(w.d, 1))} – ${dLong(addDays(w.d, 7))}${w.live ? " (live)" : ""}</b>
      <div><span>Cells that had a fire</span><span>${t.actual === null ? "open" : t.actual}</span></div>
      <div><span>Model expected</span><span>${count(t.model)}</span></div>
      <div><span>Normal</span><span>${count(t.normal)}</span></div>`;
    tip.hidden = false;
    const width = tip.offsetWidth;
    tip.style.left = `${px + 14 + width > chart.clientWidth ? px - width - 10 : px + 14}px`;
  });
  chart.addEventListener("pointerleave", () => {
    tip.hidden = true;
    chart.querySelector("#hover-line")?.setAttribute("visibility", "hidden");
  });
  chart.addEventListener("click", (evt) => { if (geom) go(nearest(evt).g); });

  // ------------------------------------------------------------------ scores table
  function drawScores() {
    const f = (v) => (v === undefined ? "–" : v.toFixed(3));
    const cellPair = (a, b) => `<td class="${a > b ? "win" : ""}">${f(a)}</td><td class="${b > a ? "win" : ""}">${f(b)}</td>`;
    const body = TARGETS.map((t) => {
      const te = S.scores[t.key];
      const se = SEA.scores[t.key] || {};
      return `<tr><td>${t.name}</td>${cellPair(te.model.roc_auc, te.climatology.roc_auc)}
        ${cellPair(se.model?.roc_auc, se.climatology?.roc_auc)}
        ${cellPair(te.spearman.model, te.spearman.climatology)}</tr>`;
    }).join("");
    const nSeason = SEA.complete.filter(Boolean).length;
    $("scores").innerHTML = `<thead>
      <tr><th></th><th colspan="2">2021–25 ROC-AUC</th><th colspan="2">${SEA.year} ROC-AUC</th><th colspan="2">Timing, 2021–25</th></tr>
      <tr><th>Target</th><th>Model</th><th>Normal</th><th>Model</th><th>Normal</th><th>Model</th><th>Normal</th></tr></thead>
      <tbody>${body}</tbody>`;
    $("updated").textContent = `${SEA.year} figures cover ${nSeason} completed week${nSeason === 1 ? "" : "s"}; data refreshed ${dLong(parse(SEA.generated.slice(0, 10)))}.`;
  }

  // ------------------------------------------------------------------ controls
  $("target").innerHTML = TARGETS.map((t) => `<button role="radio" data-v="${t.key}" aria-checked="${t.key === state.t}">${t.name}</button>`).join("");
  function seg(id, onPick) {
    $(id).addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      $(id).querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x === b)));
      onPick(b.dataset.v);
    });
  }
  seg("target", async (v) => { state.t = v; await loadTarget(v); drawLegend(); draw(); drawChart(); });
  seg("source", (v) => { state.src = v; draw(); });
  const go = (g) => { state.w = Math.max(0, Math.min(G - 1, g)); draw(); drawChart(); };
  $("prev").addEventListener("click", () => go(state.w - 1));
  $("next").addEventListener("click", () => go(state.w + 1));
  $("latest").addEventListener("click", () => go(G - 1));
  $("k").addEventListener("input", (e) => { state.k = Number(e.target.value); draw(); });
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === "ArrowLeft") go(state.w - 1);
    if (e.key === "ArrowRight") go(state.w + 1);
  });
  window.addEventListener("hashchange", async () => {     // edited links, back / forward
    const { t, g } = readHash();
    if (t && t !== state.t) {
      state.t = t;
      $("target").querySelectorAll("button").forEach((x) => x.setAttribute("aria-checked", String(x.dataset.v === t)));
      await loadTarget(t);
      drawLegend();
    }
    if (g >= 0) state.w = g;
    draw();
    drawChart();
  });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(drawChart, 120); });

  await loadTarget(state.t);
  drawLegend();
  drawScores();
  draw();
  drawChart();
})();
