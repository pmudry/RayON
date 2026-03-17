// ═══════════════════════════════════════════════════════════════════════════════
// RayON — Animated ray background for the documentation site
// ═══════════════════════════════════════════════════════════════════════════════
//
// This script renders a canvas-based ray animation behind all page content.
// It adapts to dark (slate) and light (default) MkDocs Material themes.
// A hidden control panel is accessible via a toggle button (bottom-right).
//
// To tweak the look, edit the PRESETS or DEFAULT_PRESET below.
// ═══════════════════════════════════════════════════════════════════════════════

(function () {
   "use strict";

   // ── Tunable parameters ────────────────────────────────────────────────────
   // Edit these freely to change the default appearance.

   const DEFAULT_PRESET = "subtle";

   const PRESETS = {
      subtle:   { count: 12, masterOpacity: 0.6,  speedMin: 0.1,  speedMax: 0.3,  opacityMin: 0.06, opacityMax: 0.18, glow: 0.2, dotBrightness: 0.8, rayLength: 400, fadeZone: 120 },
      balanced: { count: 20, masterOpacity: 1.0,  speedMin: 0.2,  speedMax: 0.5,  opacityMin: 0.10, opacityMax: 0.30, glow: 0.5, dotBrightness: 1.2, rayLength: 500, fadeZone: 150 },
      vivid:    { count: 30, masterOpacity: 1.4,  speedMin: 0.3,  speedMax: 0.7,  opacityMin: 0.15, opacityMax: 0.45, glow: 0.8, dotBrightness: 1.6, rayLength: 550, fadeZone: 150 },
      intense:  { count: 40, masterOpacity: 1.8,  speedMin: 0.5,  speedMax: 1.0,  opacityMin: 0.20, opacityMax: 0.55, glow: 1.0, dotBrightness: 2.0, rayLength: 600, fadeZone: 180 },
   };

   // Ray colour palettes — dark theme uses brighter tones, light uses deeper
   const RAY_COLORS_DARK = [
      { r: 133, g: 183, b: 235 },   // blue
      { r: 175, g: 169, b: 236 },   // purple
      { r: 237, g: 147, b: 177 },   // pink
      { r: 239, g: 159, b: 39  },   // amber
      { r: 93,  g: 202, b: 165 },   // teal
   ];

   const RAY_COLORS_LIGHT = [
      { r: 50,  g: 110, b: 190 },   // deeper blue
      { r: 110, g: 90,  b: 190 },   // deeper purple
      { r: 190, g: 70,  b: 120 },   // deeper pink
      { r: 190, g: 110, b: 15  },   // deeper amber
      { r: 30,  g: 150, b: 110 },   // deeper teal
   ];

   // Light theme dims rays so they don't overwhelm white backgrounds
   const LIGHT_THEME_OPACITY_FACTOR = 0.65;

   // ── Ray rendering constants ───────────────────────────────────────────────

   const SPAWN_PAD            = 20;
   const DESPAWN_MARGIN       = 900;
   const ANGLE_SPREAD         = 0.35;
   const WIDTH_MIN            = 0.8;
   const WIDTH_MAX            = 2.5;
   const RAY_TAIL_FADE        = 0.2;

   // Dots
   const DOTS_MIN             = 3;
   const DOTS_MAX             = 7;
   const DOT_RADIUS_MIN       = 1.2;
   const DOT_RADIUS_MAX       = 3.5;
   const DOT_PULSE_SPEED_MIN  = 0.02;
   const DOT_PULSE_SPEED_MAX  = 0.06;
   const DOT_SCATTER          = 3;
   const DOT_TRAVEL_SPEED_MIN = 0.0004;
   const DOT_TRAVEL_SPEED_MAX = 0.0015;

   // Spawning / stagger
   const INIT_STAGGER_MAX_FRAMES = 1200;
   const RESPAWN_DELAY_MIN       = 30;
   const RESPAWN_DELAY_MAX       = 250;

   // Tip pulsing circle
   const TIP_CIRCLE_RADIUS_MIN   = 3;
   const TIP_CIRCLE_RADIUS_MAX   = 6;
   const TIP_CIRCLE_PULSE_SPEED  = 0.025;
   const TIP_CIRCLE_GLOW_SIZE    = 10;
   const TIP_CIRCLE_MIN_OPACITY  = 0.4;
   const TIP_CIRCLE_MAX_OPACITY  = 1.0;

   // Keys that require ray re-initialisation when slider changes
   const REINIT_KEYS = new Set(["speedMin", "speedMax", "rayLength", "count"]);

   // ── Helpers ───────────────────────────────────────────────────────────────

   const rand = (a, b) => Math.random() * (b - a) + a;
   const lerp = (a, b, t) => a + (b - a) * t;

   // ── State ─────────────────────────────────────────────────────────────────

   let cfg = { ...PRESETS[DEFAULT_PRESET] };
   let rays = [];
   let activePreset = DEFAULT_PRESET;
   let W = 0, H = 0;
   let animId = null;
   let rayColors = RAY_COLORS_DARK;
   let themeOpacityFactor = 1.0;

   // ── Theme detection ───────────────────────────────────────────────────────

   function getScheme() {
      return document.body.getAttribute("data-md-color-scheme") || "slate";
   }

   function applyTheme() {
      const isLight = getScheme() === "default";
      rayColors = isLight ? RAY_COLORS_LIGHT : RAY_COLORS_DARK;
      themeOpacityFactor = isLight ? LIGHT_THEME_OPACITY_FACTOR : 1.0;
      // Re-colour existing rays
      for (const r of rays) {
         r.color = rayColors[Math.floor(Math.random() * rayColors.length)];
      }
   }

   // ── Canvas setup ──────────────────────────────────────────────────────────

   const cv  = document.getElementById("rayon-bg-canvas");
   if (!cv) return; // bail if the canvas wasn't injected
   const ctx = cv.getContext("2d");

   function resize() {
      const dpr = window.devicePixelRatio || 1;
      W = window.innerWidth;
      H = window.innerHeight;
      cv.width  = W * dpr;
      cv.height = H * dpr;
      cv.style.width  = W + "px";
      cv.style.height = H + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
   }

   // ── Ray factory ───────────────────────────────────────────────────────────

   function edgeFade(x, y, ex, ey, w, h, zone) {
      if (zone <= 0) return 1;
      const cx = (x + ex) / 2, cy = (y + ey) / 2;
      const d = Math.min(cx, w - cx, cy, h - cy);
      if (d >= zone) return 1;
      if (d <= 0) return 0;
      return d / zone;
   }

   function createRay(w, h, delay) {
      const color = rayColors[Math.floor(Math.random() * rayColors.length)];
      const side = Math.floor(Math.random() * 4);
      let tipX, tipY, angle;

      if (side === 0) {        // left
         tipX = -SPAWN_PAD; tipY = rand(0, h);
         angle = rand(-ANGLE_SPREAD, ANGLE_SPREAD);
      } else if (side === 1) { // right
         tipX = w + SPAWN_PAD; tipY = rand(0, h);
         angle = rand(Math.PI - ANGLE_SPREAD, Math.PI + ANGLE_SPREAD);
      } else if (side === 2) { // top
         tipX = rand(0, w); tipY = -SPAWN_PAD;
         angle = rand(ANGLE_SPREAD, Math.PI - ANGLE_SPREAD);
      } else {                 // bottom
         tipX = rand(0, w); tipY = h + SPAWN_PAD;
         angle = rand(-Math.PI + ANGLE_SPREAD, -ANGLE_SPREAD);
      }

      const speed  = rand(cfg.speedMin, cfg.speedMax);
      const length = rand(cfg.rayLength * 0.6, cfg.rayLength * 1.2);
      const dirX = Math.cos(angle), dirY = Math.sin(angle);

      return {
         x: tipX - dirX * length,
         y: tipY - dirY * length,
         vx: dirX * speed, vy: dirY * speed,
         dx: dirX * length, dy: dirY * length,
         length,
         baseOpacity: rand(cfg.opacityMin, cfg.opacityMax),
         width: rand(WIDTH_MIN, WIDTH_MAX),
         color,
         delay: delay || 0,
         tipPhase: rand(0, Math.PI * 2),
         dots: Array.from({ length: Math.floor(rand(DOTS_MIN, DOTS_MAX + 1)) }, () => ({
            t: rand(0, 1),
            speed: rand(DOT_TRAVEL_SPEED_MIN, DOT_TRAVEL_SPEED_MAX),
            radius: rand(DOT_RADIUS_MIN, DOT_RADIUS_MAX),
            phase: rand(0, Math.PI * 2),
            pulseSpeed: rand(DOT_PULSE_SPEED_MIN, DOT_PULSE_SPEED_MAX),
            ox: rand(-DOT_SCATTER, DOT_SCATTER),
            oy: rand(-DOT_SCATTER, DOT_SCATTER),
         })),
      };
   }

   function createInitRay(w, h) {
      const r = createRay(w, h, 0);
      const steps = rand(200, 1200);
      r.x += r.vx * steps;
      r.y += r.vy * steps;
      return r;
   }

   function initRays() {
      rays = [];
      const prePop = Math.min(Math.floor(cfg.count * 0.4), 8);
      for (let i = 0; i < prePop; i++) rays.push(createInitRay(W, H));
      for (let i = prePop; i < cfg.count; i++) {
         const base = ((i - prePop) / (cfg.count - prePop)) * INIT_STAGGER_MAX_FRAMES;
         rays.push(createRay(W, H, Math.max(0, Math.floor(base + rand(-50, 50)))));
      }
   }

   function reinitRays() {
      const old = rays.slice();
      rays = [];
      for (let i = 0; i < cfg.count; i++) {
         if (i < old.length && old[i].delay <= 0) {
            const o = old[i];
            const speed  = rand(cfg.speedMin, cfg.speedMax);
            const length = rand(cfg.rayLength * 0.6, cfg.rayLength * 1.2);
            const mag = Math.sqrt(o.vx * o.vx + o.vy * o.vy);
            const dX = mag > 0 ? o.vx / mag : 1, dY = mag > 0 ? o.vy / mag : 0;
            o.vx = dX * speed; o.vy = dY * speed;
            o.dx = dX * length; o.dy = dY * length;
            o.length = length;
            rays.push(o);
         } else {
            rays.push(createRay(W, H, Math.floor(rand(0, 60))));
         }
      }
   }

   // ── Render loop ───────────────────────────────────────────────────────────

   function makeGrad(r, ex, ey, fo, forGlow) {
      const g = ctx.createLinearGradient(r.x, r.y, ex, ey);
      const c = r.color;
      const a = forGlow ? fo * cfg.glow * 0.3 : fo;
      g.addColorStop(0,             `rgba(${c.r},${c.g},${c.b},0)`);
      g.addColorStop(RAY_TAIL_FADE, `rgba(${c.r},${c.g},${c.b},${a})`);
      g.addColorStop(1,             `rgba(${c.r},${c.g},${c.b},${a})`);
      return g;
   }

   function tick() {
      ctx.clearRect(0, 0, W, H);

      while (rays.length < cfg.count)
         rays.push(createRay(W, H, Math.floor(rand(RESPAWN_DELAY_MIN, RESPAWN_DELAY_MAX))));
      if (rays.length > cfg.count) rays.length = cfg.count;

      for (let i = 0; i < rays.length; i++) {
         const r = rays[i];
         if (r.delay > 0) { r.delay--; continue; }

         r.x += r.vx; r.y += r.vy;
         const ex = r.x + r.dx, ey = r.y + r.dy;

         // Despawn
         const xMin = Math.min(r.x, ex), xMax = Math.max(r.x, ex);
         const yMin = Math.min(r.y, ey), yMax = Math.max(r.y, ey);
         if (xMax < -DESPAWN_MARGIN || xMin > W + DESPAWN_MARGIN ||
             yMax < -DESPAWN_MARGIN || yMin > H + DESPAWN_MARGIN) {
            rays[i] = createRay(W, H, Math.floor(rand(RESPAWN_DELAY_MIN, RESPAWN_DELAY_MAX)));
            continue;
         }

         const ef = edgeFade(r.x, r.y, ex, ey, W, H, cfg.fadeZone);
         const fo = r.baseOpacity * ef * cfg.masterOpacity * themeOpacityFactor;
         if (fo < 0.003) continue;

         // Glow
         if (cfg.glow > 0) {
            ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(ex, ey);
            ctx.strokeStyle = makeGrad(r, ex, ey, fo, true);
            ctx.lineWidth = r.width + 6; ctx.lineCap = "round"; ctx.stroke();
         }

         // Main ray
         ctx.beginPath(); ctx.moveTo(r.x, r.y); ctx.lineTo(ex, ey);
         ctx.strokeStyle = makeGrad(r, ex, ey, fo, false);
         ctx.lineWidth = r.width; ctx.lineCap = "round"; ctx.stroke();

         // Tip circle
         r.tipPhase += TIP_CIRCLE_PULSE_SPEED;
         const pulse = 0.5 + 0.5 * Math.sin(r.tipPhase);
         const tipR  = lerp(TIP_CIRCLE_RADIUS_MIN, TIP_CIRCLE_RADIUS_MAX, pulse);
         const tipA  = lerp(TIP_CIRCLE_MIN_OPACITY, TIP_CIRCLE_MAX_OPACITY, pulse) * fo;

         ctx.beginPath(); ctx.arc(ex, ey, tipR + TIP_CIRCLE_GLOW_SIZE, 0, Math.PI * 2);
         ctx.fillStyle = `rgba(${r.color.r},${r.color.g},${r.color.b},${tipA * 0.12})`;
         ctx.fill();
         ctx.beginPath(); ctx.arc(ex, ey, tipR + TIP_CIRCLE_GLOW_SIZE * 0.4, 0, Math.PI * 2);
         ctx.fillStyle = `rgba(${r.color.r},${r.color.g},${r.color.b},${tipA * 0.25})`;
         ctx.fill();
         ctx.beginPath(); ctx.arc(ex, ey, tipR, 0, Math.PI * 2);
         // Dark theme: white core; Light theme: dark core so it's visible
         var tipCore = themeOpacityFactor < 1 ? `rgba(30,20,60,${tipA})` : `rgba(255,255,255,${tipA})`;
         ctx.fillStyle = tipCore;
         ctx.fill();

         // Dots
         for (const d of r.dots) {
            d.t -= d.speed;
            if (d.t < 0) d.t += 1;
            const dotFade = d.t < RAY_TAIL_FADE ? d.t / RAY_TAIL_FADE : 1;
            const px = lerp(r.x, ex, d.t) + d.ox;
            const py = lerp(r.y, ey, d.t) + d.oy;
            d.phase += d.pulseSpeed;
            const dp  = 0.5 + 0.5 * Math.sin(d.phase);
            const dop = fo * (0.5 + 0.5 * dp) * cfg.dotBrightness * dotFade;
            const dr  = d.radius * (0.6 + 0.4 * dp);

            if (cfg.glow > 0 && dop > 0.01) {
               ctx.beginPath(); ctx.arc(px, py, dr + 3, 0, Math.PI * 2);
               ctx.fillStyle = `rgba(${r.color.r},${r.color.g},${r.color.b},${dop * cfg.glow * 0.25})`;
               ctx.fill();
            }
            if (dop > 0.005) {
               ctx.beginPath(); ctx.arc(px, py, dr, 0, Math.PI * 2);
               ctx.fillStyle = `rgba(${r.color.r},${r.color.g},${r.color.b},${dop})`;
               ctx.fill();
            }
         }
      }

      animId = requestAnimationFrame(tick);
   }

   // ── Visibility API — pause when tab is hidden ─────────────────────────────

   document.addEventListener("visibilitychange", function () {
      if (document.hidden) {
         if (animId) { cancelAnimationFrame(animId); animId = null; }
      } else {
         if (!animId) animId = requestAnimationFrame(tick);
      }
   });

   // ── Reduced-motion preference ─────────────────────────────────────────────

   if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      cv.style.display = "none";
      return; // skip everything
   }

   // ── Control panel (hidden sidebar) ────────────────────────────────────────

   function buildControlPanel() {
      const sidebar = document.getElementById("rayon-bg-sidebar");
      const toggle  = document.getElementById("rayon-bg-toggle");
      if (!sidebar || !toggle) return;

      // Toggle sidebar open/close; right-click toggles animation on/off
      toggle.addEventListener("click", function () {
         sidebar.classList.toggle("open");
      });
      toggle.addEventListener("contextmenu", function (e) {
         e.preventDefault();
         toggleBackground();
      });

      // Title
      const title = document.createElement("div");
      title.className = "rayon-panel-title";
      title.textContent = "Ray controls";
      sidebar.appendChild(title);

      // Presets
      const preLabel = document.createElement("div");
      preLabel.className = "rayon-section-label";
      preLabel.textContent = "Presets";
      sidebar.appendChild(preLabel);

      const presetWrap = document.createElement("div");
      presetWrap.className = "rayon-presets";
      sidebar.appendChild(presetWrap);

      const presetBtns = {};

      function updatePresetBtns() {
         Object.entries(presetBtns).forEach(function (entry) {
            entry[1].className = "rayon-preset-btn" + (entry[0] === activePreset ? " active" : "");
         });
      }

      function applyPreset(name) {
         cfg = { ...PRESETS[name] };
         activePreset = name;
         sliderDefs.forEach(function (s) {
            var inp = document.getElementById("rayon_sl_" + s.key);
            if (inp) inp.value = cfg[s.key];
            var v = valSpans[s.key];
            if (v) v.span.textContent = (v.step < 1 ? cfg[s.key].toFixed(2) : cfg[s.key]) + v.unit;
         });
         updatePresetBtns();
         reinitRays();
      }

      Object.keys(PRESETS).forEach(function (name) {
         var btn = document.createElement("button");
         btn.textContent = name;
         btn.className = "rayon-preset-btn";
         btn.addEventListener("click", function () { applyPreset(name); });
         presetBtns[name] = btn;
         presetWrap.appendChild(btn);
      });
      updatePresetBtns();

      // Separator
      var sep1 = document.createElement("div");
      sep1.className = "rayon-sep";
      sidebar.appendChild(sep1);

      // Sliders
      var sliderDefs = [
         { key: "count",         label: "Ray count",      min: 4,    max: 60,  step: 1,    unit: "" },
         { key: "masterOpacity", label: "Intensity",       min: 0.1,  max: 2.5, step: 0.05, unit: "" },
         { key: "opacityMin",    label: "Min opacity",     min: 0.02, max: 0.4, step: 0.01, unit: "" },
         { key: "opacityMax",    label: "Max opacity",     min: 0.1,  max: 0.8, step: 0.01, unit: "" },
         { key: "speedMin",      label: "Min speed",       min: 0.05, max: 1.0, step: 0.05, unit: " px/f" },
         { key: "speedMax",      label: "Max speed",       min: 0.1,  max: 2.0, step: 0.05, unit: " px/f" },
         { key: "rayLength",     label: "Ray length",      min: 100,  max: 800, step: 10,   unit: " px" },
         { key: "glow",          label: "Glow",            min: 0,    max: 1.5, step: 0.05, unit: "" },
         { key: "dotBrightness", label: "Dot brightness",  min: 0,    max: 3.0, step: 0.1,  unit: "" },
         { key: "fadeZone",      label: "Edge fade zone",  min: 0,    max: 300, step: 10,   unit: " px" },
      ];
      var valSpans = {};

      sliderDefs.forEach(function (s) {
         var row  = document.createElement("div"); row.className = "rayon-slider-row";
         var head = document.createElement("div"); head.className = "rayon-slider-head";
         var lbl  = document.createElement("span"); lbl.className = "rayon-slider-label"; lbl.textContent = s.label;
         var val  = document.createElement("span"); val.className = "rayon-slider-val";
         val.textContent = (s.step < 1 ? cfg[s.key].toFixed(2) : cfg[s.key]) + s.unit;
         valSpans[s.key] = { span: val, unit: s.unit, step: s.step };
         head.appendChild(lbl); head.appendChild(val);

         var inp = document.createElement("input");
         inp.type = "range"; inp.min = s.min; inp.max = s.max; inp.step = s.step;
         inp.value = cfg[s.key]; inp.id = "rayon_sl_" + s.key;
         inp.addEventListener("input", function () {
            cfg[s.key] = Number(inp.value);
            val.textContent = (s.step < 1 ? cfg[s.key].toFixed(2) : cfg[s.key]) + s.unit;
            activePreset = null;
            updatePresetBtns();
            if (REINIT_KEYS.has(s.key)) reinitRays();
         });

         row.appendChild(head); row.appendChild(inp);
         sidebar.appendChild(row);
      });

      // Separator + footer
      var sep2 = document.createElement("div");
      sep2.className = "rayon-sep";
      sidebar.appendChild(sep2);

      var footer = document.createElement("p");
      footer.className = "rayon-footer-text";
      footer.textContent = "Rays enter from outside the viewport, travel straight across, and exit the other side. A pulsing circle marks each leading tip.";
      sidebar.appendChild(footer);
   }

   // ── Toggle animation on/off ───────────────────────────────────────────────

   let bgEnabled = true;

   function toggleBackground() {
      bgEnabled = !bgEnabled;
      if (bgEnabled) {
         cv.style.display = "";
         if (!animId) animId = requestAnimationFrame(tick);
      } else {
         cv.style.display = "none";
         if (animId) { cancelAnimationFrame(animId); animId = null; }
      }
      // Update toggle button icon
      var toggle = document.getElementById("rayon-bg-toggle");
      if (toggle) toggle.textContent = bgEnabled ? "\u2699" : "\u25CB";
   }

   // Keyboard shortcut: Shift+B toggles background
   document.addEventListener("keydown", function (e) {
      if (e.shiftKey && e.key === "B" && !e.ctrlKey && !e.altKey && !e.metaKey) {
         // Don't trigger if user is typing in an input/textarea
         var tag = (e.target.tagName || "").toLowerCase();
         if (tag === "input" || tag === "textarea" || e.target.isContentEditable) return;
         e.preventDefault();
         toggleBackground();
      }
   });

   // ── Bootstrap ─────────────────────────────────────────────────────────────

   applyTheme();
   resize();
   initRays();
   animId = requestAnimationFrame(tick);
   window.addEventListener("resize", resize);
   buildControlPanel();

   // Watch for theme toggles
   var obs = new MutationObserver(function (mutations) {
      for (var i = 0; i < mutations.length; i++) {
         if (mutations[i].attributeName === "data-md-color-scheme") {
            applyTheme();
            break;
         }
      }
   });
   obs.observe(document.body, { attributes: true });

})();
