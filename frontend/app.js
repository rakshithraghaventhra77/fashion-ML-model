const state = {
  file: null,
  previewUrl: null,
  results: [],
  loading: false,
  error: "",
};

const API_URL = "http://localhost:8080/api/recommend";

const elements = {
  form: document.getElementById("uploadForm"),
  fileInput: document.getElementById("fileInput"),
  dropzone: document.getElementById("dropzone"),
  previewImage: document.getElementById("previewImage"),
  previewEmpty: document.getElementById("previewEmpty"),
  fileName: document.getElementById("fileName"),
  submitBtn: document.getElementById("submitBtn"),
  clearBtn: document.getElementById("clearBtn"),
  resultsGrid: document.getElementById("resultsGrid"),
  emptyState: document.getElementById("emptyState"),
  loadingState: document.getElementById("loadingState"),
  errorState: document.getElementById("errorState"),
  resultSummary: document.getElementById("resultSummary"),
  resultTemplate: document.getElementById("resultCardTemplate"),
};

function showElement(element, display = "block") {
  element.hidden = false;
  element.style.display = display;
}

function hideElement(element) {
  element.hidden = true;
  element.style.display = "none";
}

function showError(message) {
  showElement(elements.errorState, "block");
  elements.errorState.textContent = message;
  state.error = message;
}

function clearError() {
  state.error = "";
  hideElement(elements.errorState);
  elements.errorState.textContent = "";
}

function setLoading(isLoading) {
  state.loading = isLoading;
  if (isLoading) {
    showElement(elements.loadingState, "flex");
    hideElement(elements.emptyState);
  } else {
    hideElement(elements.loadingState);
  }
  elements.submitBtn.disabled = isLoading;
  elements.submitBtn.textContent = isLoading ? "Analyzing..." : "Analyze image";
}

function setPreview(file) {
  state.file = file;
  elements.fileName.textContent = file ? file.name : "No file selected";

  if (state.previewUrl) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
  }

  if (file) {
    state.previewUrl = URL.createObjectURL(file);
    elements.previewImage.src = state.previewUrl;
    showElement(elements.previewImage, "block");
    hideElement(elements.previewEmpty);
    clearError();
  } else {
    elements.previewImage.removeAttribute("src");
    hideElement(elements.previewImage);
    showElement(elements.previewEmpty, "grid");
  }
}

function resetResults() {
  state.results = [];
  elements.resultsGrid.innerHTML = "";
  hideElement(elements.loadingState);
  showElement(elements.emptyState, "grid");
  elements.resultSummary.textContent = "No analysis run yet";
}

function safeParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return { raw: text };
  }
}

function buildImageUrl(rec) {
  if (rec.imageUrl) return rec.imageUrl;
  return "";
}

function createTag(label) {
  const tag = document.createElement("span");
  tag.className = "tag";
  tag.textContent = label;
  return tag;
}

function renderResults(results) {
  elements.resultsGrid.innerHTML = "";

  if (!Array.isArray(results) || results.length === 0) {
    hideElement(elements.loadingState);
    showElement(elements.emptyState, "grid");
    elements.resultSummary.textContent = "No recommendations returned";
    return;
  }

  hideElement(elements.loadingState);
  hideElement(elements.emptyState);
  elements.resultSummary.textContent = `${results.length} recommendation${results.length === 1 ? "" : "s"} found`;

  results.forEach((rec, index) => {
    const card = elements.resultTemplate.content.cloneNode(true);
    const root = card.querySelector(".result-card");
    const image = card.querySelector(".result-image");
    const title = card.querySelector(".result-title");
    const id = card.querySelector(".result-id");
    const meta = card.querySelector(".result-meta");
    const name = card.querySelector(".result-name");
    const tags = card.querySelector(".result-tags");

    const imageUrl = buildImageUrl(rec);
    image.src = imageUrl || "";
    image.alt = rec.productDisplayName || rec.articleType || `Recommendation ${index + 1}`;
    image.onerror = () => {
      image.style.objectFit = "contain";
      image.src = "data:image/svg+xml;charset=UTF-8," + encodeURIComponent(`
        <svg xmlns="http://www.w3.org/2000/svg" width="800" height="900" viewBox="0 0 800 900">
          <rect width="800" height="900" fill="#0c1725"/>
          <rect x="110" y="140" width="580" height="620" rx="36" fill="#152333" stroke="#2c3d50"/>
          <text x="50%" y="50%" fill="#7d8ea1" font-family="Arial" font-size="34" text-anchor="middle">Image unavailable</text>
        </svg>
      `);
    };

    title.textContent = rec.articleType || rec.masterCategory || "Recommendation";
    id.textContent = `#${rec.id || index + 1}`;
    meta.textContent = [rec.gender, rec.masterCategory, rec.baseColour, rec.season].filter(Boolean).join(" • ");
    name.textContent = rec.productDisplayName || "No product name available";

    const tagValues = [rec.subCategory, rec.usage, rec.year && `Year ${rec.year}`].filter(Boolean);
    tagValues.forEach((value) => tags.appendChild(createTag(value)));

    root.querySelector(".result-image-wrap").appendChild(image);
    elements.resultsGrid.appendChild(card);
  });
}

async function handleSubmit(event) {
  event.preventDefault();
  clearError();

  if (!state.file) {
    showError("Choose an image first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", state.file);

  try {
    setLoading(true);
    const response = await fetch(API_URL, {
      method: "POST",
      body: formData,
    });

    const text = await response.text();
    const payload = safeParse(text);

    if (!response.ok) {
      const message = payload.detail || payload.error || payload.message || "Request failed";
      throw new Error(message);
    }

    const results = payload.recommendations || [];
    state.results = results;
    renderResults(results);
  } catch (error) {
    showError(error.message || "Something went wrong while fetching recommendations.");
  } finally {
    setLoading(false);
  }
}

function clearSelection() {
  elements.fileInput.value = "";
  setPreview(null);
  clearError();
  resetResults();
}

function handleDrop(event) {
  event.preventDefault();
  elements.dropzone.classList.remove("dragover");

  const file = event.dataTransfer.files?.[0];
  if (file && file.type.startsWith("image/")) {
    setPreview(file);
  }
}

function handleDragOver(event) {
  event.preventDefault();
  elements.dropzone.classList.add("dragover");
}

function handleDragLeave() {
  elements.dropzone.classList.remove("dragover");
}

function init() {
  resetResults();
  clearError();

  elements.form.addEventListener("submit", handleSubmit);
  elements.fileInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0] || null;
    if (file && file.type.startsWith("image/")) {
      setPreview(file);
    }
  });

  elements.clearBtn.addEventListener("click", clearSelection);
  elements.dropzone.addEventListener("dragover", handleDragOver);
  elements.dropzone.addEventListener("dragleave", handleDragLeave);
  elements.dropzone.addEventListener("drop", handleDrop);
}

init();
