const state = {
  results: [],
  selectedAssetId: null,
  mode: "idle",
};

const resultsGrid = document.getElementById("results-grid");
const detailPanel = document.getElementById("detail-panel");
const resultsMeta = document.getElementById("results-meta");
const statsEl = document.getElementById("stats");
const folderFilter = document.getElementById("folder-filter");
const contentTypeFilter = document.getElementById("content-type-filter");
const searchForm = document.getElementById("search-form");
const queryInput = document.getElementById("query");
const uploadInput = document.getElementById("image-upload");
const uploadTrigger = document.getElementById("upload-trigger");
const dropzone = document.getElementById("dropzone");
const template = document.getElementById("result-card-template");

function basename(filePath) {
  return (filePath || "").split("/").filter(Boolean).pop() || "Image";
}

function makeBadges(result) {
  const items = [
    result.content_type,
    ...(result.style_labels || []).slice(0, 2),
    ...(result.auto_tags || []).slice(0, 2),
  ].filter(Boolean);
  return items;
}

function compactFolder(filePath) {
  const parts = (filePath || "").split("/").filter(Boolean);
  if (parts.length <= 2) {
    return filePath || "Unknown folder";
  }
  return parts.slice(-3, -1).join("/");
}

function formatDate(value) {
  if (!value) {
    return "";
  }
  return value.slice(0, 10);
}

function renderBadgeList(labels, className = "badge") {
  return (labels || [])
    .filter(Boolean)
    .slice(0, 8)
    .map((label) => `<span class="${className}">${label}</span>`)
    .join("");
}

function setResults(results, label) {
  state.results = results;
  resultsMeta.textContent = `${results.length} result${results.length === 1 ? "" : "s"}${label ? ` for ${label}` : ""}`;
  renderResults();
  if (results[0]) {
    openAsset(results[0].id);
  } else {
    detailPanel.innerHTML = '<div class="detail-empty">No results yet. Try a broader query or a different image.</div>';
  }
}

function renderResults() {
  resultsGrid.innerHTML = "";
  if (!state.results.length) {
    resultsGrid.innerHTML = '<div class="empty-state">Run a search or drop an image to begin.</div>';
    return;
  }

  state.results.forEach((result) => {
    const fragment = template.content.cloneNode(true);
    const card = fragment.querySelector(".result-card");
    const image = fragment.querySelector(".result-thumb");
    const title = fragment.querySelector(".result-title");
    const meta = fragment.querySelector(".result-meta");
    const badgeRow = fragment.querySelector(".badge-row");

    image.src = result.thumbnail_url;
    image.alt = result.summary || result.file_path;
    title.textContent = basename(result.file_path);
    meta.textContent = `${result.folder || "Unknown folder"}${result.modified_at ? ` • ${result.modified_at.slice(0, 10)}` : ""}`;
    badgeRow.innerHTML = "";
    makeBadges(result).forEach((label) => {
      const badge = document.createElement("span");
      badge.className = "badge";
      badge.textContent = label;
      badgeRow.appendChild(badge);
    });
    card.addEventListener("click", () => openAsset(result.id));
    resultsGrid.appendChild(fragment);
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(payload.detail || "Request failed");
  }
  return response.json();
}

async function refreshStats() {
  const payload = await fetchJson("/stats");
  statsEl.textContent = `${payload.asset_count} indexed assets • ${payload.indexed_paths.length} folders`;
}

async function loadFolders() {
  const payload = await fetchJson("/folders");
  folderFilter.innerHTML = '<option value="">All indexed folders</option>';
  payload.folders.forEach((folder) => {
    const option = document.createElement("option");
    option.value = folder;
    option.textContent = folder;
    folderFilter.appendChild(option);
  });
}

async function loadContentTypes() {
  const payload = await fetchJson("/content-types");
  contentTypeFilter.innerHTML = '<option value="">All content types</option>';
  payload.content_types.forEach((contentType) => {
    const option = document.createElement("option");
    option.value = contentType;
    option.textContent = contentType;
    contentTypeFilter.appendChild(option);
  });
}

async function searchText(event) {
  event.preventDefault();
  const query = queryInput.value.trim();
  if (!query) {
    return;
  }

  resultsMeta.textContent = "Searching...";
  const payload = {
    query,
    folder: folderFilter.value || null,
    content_type: contentTypeFilter.value || null,
    date_from: document.getElementById("date-from").value || null,
    date_to: document.getElementById("date-to").value || null,
    top_k: 36,
  };
  const response = await fetchJson("/search/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  state.mode = "text";
  setResults(response.results, `"${query}"`);
}

async function searchSimilarFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", "36");
  resultsMeta.textContent = "Finding similar images...";
  const response = await fetchJson("/search/similar", {
    method: "POST",
    body: formData,
  });
  state.mode = "similar";
  setResults(response.results, `uploaded image "${file.name}"`);
}

async function openAsset(assetId) {
  state.selectedAssetId = assetId;
  const asset = await fetchJson(`/assets/${assetId}`);
  const labels = [
    asset.content_type,
    ...(asset.style_labels || []),
    ...(asset.auto_tags || []),
    ...(asset.object_labels || []).slice(0, 3),
  ].filter(Boolean);
  const compactMeta = [
    compactFolder(asset.file_path),
    formatDate(asset.modified_at),
    asset.width && asset.height ? `${asset.width} x ${asset.height}` : "",
  ].filter(Boolean);
  detailPanel.innerHTML = `
    <img src="${asset.image_url}" alt="${asset.summary || "Selected image"}" />
    <h3>${basename(asset.file_path)}</h3>
    <p class="detail-subtitle">${compactMeta.join(" • ")}</p>
    <div class="detail-badges">${renderBadgeList(labels)}</div>
    <div class="detail-actions">
      <button id="find-similar-button" type="button">Find similar</button>
      <a href="${asset.image_url}" target="_blank" rel="noreferrer">Open original</a>
    </div>
  `;
  const button = document.getElementById("find-similar-button");
  button.addEventListener("click", () => searchSimilarAsset(assetId));
}

async function searchSimilarAsset(assetId) {
  const formData = new FormData();
  formData.append("asset_id", String(assetId));
  formData.append("top_k", "36");
  resultsMeta.textContent = "Finding similar images...";
  const response = await fetchJson("/search/similar", {
    method: "POST",
    body: formData,
  });
  state.mode = "similar";
  setResults(response.results, `asset ${assetId}`);
}

function setupUploadInteractions() {
  uploadTrigger.addEventListener("click", () => uploadInput.click());
  uploadInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) {
      searchSimilarFile(file).catch(showError);
      uploadInput.value = "";
    }
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("drag-over");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("drag-over");
    });
  });

  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      searchSimilarFile(file).catch(showError);
    }
  });
}

function showError(error) {
  resultsMeta.textContent = error.message;
}

async function bootstrap() {
  try {
    await Promise.all([refreshStats(), loadFolders(), loadContentTypes()]);
    renderResults();
    setupUploadInteractions();
    searchForm.addEventListener("submit", (event) => {
      searchText(event).catch(showError);
    });
  } catch (error) {
    showError(error);
  }
}

bootstrap();
