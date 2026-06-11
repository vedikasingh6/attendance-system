// ============================================================
//  Attendance System – Frontend JS
// ============================================================

/* ── Alert auto-dismiss ── */
document.querySelectorAll(".alert").forEach(el => {
  setTimeout(() => {
    el.style.transition = "opacity .4s";
    el.style.opacity = "0";
    setTimeout(() => el.remove(), 400);
  }, 4000);
});

/* ── Drag-and-drop upload zones ── */
document.querySelectorAll(".upload-zone").forEach(zone => {
  const input = zone.querySelector("input[type=file]");
  if (!input) return;

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", e => {
    e.preventDefault();
    zone.classList.add("dragover");
  });

  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));

  zone.addEventListener("drop", e => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      // DataTransfer items → new FileList approach
      input.files = e.dataTransfer.files;
      updateFileLabel(zone, input);
    }
  });

  input.addEventListener("change", () => updateFileLabel(zone, input));
});

function updateFileLabel(zone, input) {
  const p = zone.querySelector("p");
  if (!p) return;
  if (input.files.length === 1) {
    p.textContent = input.files[0].name;
  } else if (input.files.length > 1) {
    p.textContent = `${input.files.length} files selected`;
  }
}

/* ── Progress ring animation ── */
function animateRing() {
  document.querySelectorAll(".ring-fill").forEach(circle => {
    const pct = parseFloat(circle.dataset.pct || 0);
    const r   = parseFloat(circle.getAttribute("r"));
    const c   = 2 * Math.PI * r;
    circle.style.strokeDasharray  = c;
    circle.style.strokeDashoffset = c - (pct / 100) * c;
  });
}
document.addEventListener("DOMContentLoaded", animateRing);

/* ── Date picker default to today ── */
document.querySelectorAll("input[type=date]").forEach(el => {
  if (!el.value) {
    const now = new Date();
    const y   = now.getFullYear();
    const m   = String(now.getMonth() + 1).padStart(2, "0");
    const d   = String(now.getDate()).padStart(2, "0");
    el.value  = `${y}-${m}-${d}`;
  }
});

/* ── Confirm dangerous actions ── */
document.querySelectorAll("[data-confirm]").forEach(el => {
  el.addEventListener("click", e => {
    if (!confirm(el.dataset.confirm)) e.preventDefault();
  });
});

/* ── Preview face images before upload ── */
const faceInput = document.getElementById("face_images");
if (faceInput) {
  faceInput.addEventListener("change", () => {
    const container = document.getElementById("face_preview");
    if (!container) return;
    container.innerHTML = "";
    Array.from(faceInput.files).slice(0, 5).forEach(file => {
      const reader = new FileReader();
      reader.onload = e => {
        const img = document.createElement("img");
        img.src   = e.target.result;
        img.style.cssText = "width:64px;height:64px;object-fit:cover;border-radius:8px;border:2px solid var(--primary);";
        container.appendChild(img);
      };
      reader.readAsDataURL(file);
    });
  });
}
