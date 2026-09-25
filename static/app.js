/* Book Reader inspect UI — project list + detail (movie, audiobook,
   characters/voices, storyboard filmstrip). Vanilla JS, no framework. */

const $ = (sel) => document.querySelector(sel);

function fmtTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

async function loadProjects() {
  const res = await fetch("/api/projects");
  const projects = await res.json();
  const host = $("#projects");
  host.innerHTML = "";
  for (const p of projects) {
    const card = el("button", "project-card");
    card.append(el("div", "name", p.name));
    const meta = el("div", "meta");
    meta.append(el("span", "", `${p.steps_done.length}/${p.steps_total} steps`));
    if (p.has_audiobook) meta.append(el("span", "", "🎧 audiobook"));
    if (p.has_movie) meta.append(el("span", "", "🎬 movie"));
    if (p.scenes) meta.append(el("span", "", `${p.scenes} scenes`));
    card.append(meta);
    const track = el("div", "progress-track");
    const fill = el("div", "progress-fill");
    fill.style.width = `${Math.round((100 * p.steps_done.length) / p.steps_total)}%`;
    track.append(fill);
    card.append(track);
    card.addEventListener("click", () => {
      document.querySelectorAll(".project-card").forEach((c) => c.classList.remove("active"));
      card.classList.add("active");
      loadDetail(p.name);
    });
    host.append(card);
  }
}

function renderSteps(detail) {
  const wrap = el("div", "step-pills");
  const steps = [
    "extract", "characters", "voices_desc", "voices_clone",
    "scripts", "audio", "m4b", "storyboard", "refimages", "sceneimages", "movie",
  ];
  for (const s of steps) {
    const done = detail.steps_done.includes(s);
    const pill = el("span", "pill" + (done ? " done" : s === detail.next_step ? " next" : ""), s);
    if (detail.timings && detail.timings[s]) pill.title = `${detail.timings[s].toFixed(1)}s`;
    wrap.append(pill);
  }
  return wrap;
}

function openLightbox(src) {
  const box = el("div", "");
  box.id = "lightbox";
  const img = el("img");
  img.src = src;
  box.append(img);
  box.addEventListener("click", () => box.remove());
  document.body.append(box);
}

async function loadDetail(name) {
  const res = await fetch(`/api/project/${encodeURIComponent(name)}`);
  const d = await res.json();
  $("#empty-state").hidden = true;
  const host = $("#project-detail");
  host.hidden = false;
  $("#d-title").textContent = d.name;

  const progress = $("#d-progress");
  progress.innerHTML = "";
  progress.append(renderSteps(d));

  const movieHost = $("#d-movie");
  movieHost.innerHTML = "";
  if (d.movie_url) {
    const block = el("div", "media-block");
    block.append(el("h3", "", "🎬 Movie"));
    const video = el("video");
    video.controls = true;
    video.preload = "metadata";
    video.src = d.movie_url;
    block.append(video);
    movieHost.append(block);
  }

  const audioHost = $("#d-audiobook");
  audioHost.innerHTML = "";
  if (d.audiobook_url) {
    const block = el("div", "media-block");
    block.append(el("h3", "", "🎧 Audiobook (M4B)"));
    const audio = el("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.src = d.audiobook_url;
    block.append(audio);
    audioHost.append(block);
  }

  const charHost = $("#d-characters");
  charHost.innerHTML = "";
  for (const c of d.characters || []) {
    const card = el("div", "character-card");
    if (c.ref_image) {
      const img = el("img");
      img.src = c.ref_image;
      img.alt = c.name;
      img.loading = "lazy";
      img.addEventListener("click", () => openLightbox(c.ref_image));
      card.append(img);
    }
    const body = el("div", "body");
    body.append(el("div", "cname", c.name));
    if (c.bio) body.append(el("div", "cbio", c.bio));
    if (c.voice_clip) {
      const audio = el("audio");
      audio.controls = true;
      audio.preload = "none";
      audio.src = c.voice_clip;
      body.append(audio);
    }
    card.append(body);
    charHost.append(card);
  }

  $("#d-style").textContent = d.style ? `Style: ${d.style}` : "";
  const sceneHost = $("#d-scenes");
  sceneHost.innerHTML = "";
  for (const s of d.scenes || []) {
    const card = el("div", "scene-card");
    if (s.image) {
      const img = el("img");
      img.src = s.image;
      img.alt = `scene ${s.index}`;
      img.loading = "lazy";
      img.addEventListener("click", () => openLightbox(s.image));
      card.append(img);
    }
    const body = el("div", "body");
    body.append(el("div", "time", `${fmtTime(s.start)} – ${fmtTime(s.end)}`));
    if (s.text_excerpt) body.append(el("div", "excerpt", s.text_excerpt.slice(0, 110) + "…"));
    if (s.characters && s.characters.length) body.append(el("div", "chars", s.characters.join(", ")));
    card.append(body);
    sceneHost.append(card);
  }
}

loadProjects().catch((e) => {
  $("#projects").append(el("p", "", `Failed to load projects: ${e}`));
});
