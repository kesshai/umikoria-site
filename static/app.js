const form = document.getElementById("pixelForm");
const inputs = Array.from(form.elements);
const fileInput = document.getElementById("imageInput");
const settingsInputs = inputs.filter((el) => el !== fileInput);
const origImg = document.getElementById("origImg");
const midImg = document.getElementById("midImg");
const afterImg = document.getElementById("afterImg");
const zoomSlider = document.getElementById("zoomSlider");
const errorBox = document.getElementById("errorBox");
const downloadLink = document.getElementById("downloadLink");
const previewBoxes = document.querySelectorAll(".preview-box");
const paletteSelect = document.getElementById("paletteSelect");
const resetLink = document.getElementById("resetLink");

let dragImg = null;
let startX = 0;
let startY = 0;

function toggleSettings(enabled) {
   settingsInputs.forEach((el) => (el.disabled = !enabled));
   zoomSlider.disabled = !enabled;
}

toggleSettings(false);

fetch("/static/palettes.json")
   .then((r) => r.json())
   .then((list) => {
      const def = document.createElement("option");
      def.value = "";
      def.textContent = "Auto";
      paletteSelect.appendChild(def);
      list.forEach((p, idx) => {
         const opt = document.createElement("option");
         opt.value = idx;
         opt.textContent = p.name;
         paletteSelect.appendChild(opt);
      });
   });

async function updatePreview() {
   if (!fileInput.files.length) {
      return;
   }
   errorBox.style.display = "none";
   const data = new FormData(form);

   try {
      const resp = await fetch("/", {
         method: "POST",
         body: data,
         headers: { "X-Requested-With": "XMLHttpRequest" },
      });
      const json = await resp.json();

      if (json.error) {
         errorBox.innerText = json.error;
         errorBox.style.display = "block";
         return;
      }

      origImg.src = json.orig;
      midImg.src = json.mid;
      afterImg.src = json.after;
      downloadLink.href = json.after;
      applyZoom();
   } catch (e) {
      errorBox.innerText = "Сетевая ошибка или сервер недоступен";
      errorBox.style.display = "block";
   }
}

settingsInputs.forEach((inp) => {
   inp.addEventListener("change", updatePreview);
   inp.addEventListener("input", updatePreview);
});
fileInput.addEventListener("change", () => {
   const hasFile = fileInput.files && fileInput.files.length > 0;
   toggleSettings(hasFile);
   if (hasFile) {
      updatePreview();
   } else {
      origImg.src = "";
      midImg.src = "";
      afterImg.src = "";
      downloadLink.href = "#";
   }
});
resetLink.addEventListener("click", (e) => {
   e.preventDefault();
   form.reset();
   fileInput.value = "";
   toggleSettings(false);
   errorBox.style.display = "none";
   downloadLink.href = "#";
   zoomSlider.value = 1;
   [origImg, midImg, afterImg].forEach((img) => {
      img.src = "";
      img._zoom = 1;
      img._panX = 0;
      img._panY = 0;
      updateTransform(img);
   });
   previewBoxes.forEach((b) => b.classList.remove("active"));
   if (previewBoxes.length) {
      previewBoxes[0].classList.add("active");
   }
});
zoomSlider.addEventListener("input", applyZoom);

[origImg, midImg, afterImg].forEach((img) => {
   img._zoom = 1;
   img._panX = 0;
   img._panY = 0;
   const wrapper = img.parentElement;
   wrapper.addEventListener("mousedown", (e) => startDrag(e, img));
});

previewBoxes.forEach((box) => {
   box.addEventListener("click", () => {
      previewBoxes.forEach((b) => b.classList.remove("active"));
      box.classList.add("active");
   });
});
if (previewBoxes.length) {
   previewBoxes[0].classList.add("active");
}

function applyZoom() {
   const z = parseFloat(zoomSlider.value);
   [origImg, midImg, afterImg].forEach((img) => {
      img._zoom = z;
      updateTransform(img);
   });
}

function startDrag(e, img) {
   dragImg = img;
   startX = e.clientX;
   startY = e.clientY;
   img.parentElement.style.cursor = "grabbing";
   document.addEventListener("mousemove", onDrag);
   document.addEventListener("mouseup", endDrag);
}

function onDrag(e) {
   if (!dragImg) return;
   dragImg._panX += e.clientX - startX;
   dragImg._panY += e.clientY - startY;
   startX = e.clientX;
   startY = e.clientY;
   updateTransform(dragImg);
}

function endDrag() {
   if (dragImg) {
      dragImg.parentElement.style.cursor = "grab";
   }
   document.removeEventListener("mousemove", onDrag);
   document.removeEventListener("mouseup", endDrag);
   dragImg = null;
}

function updateTransform(img) {
   img.style.transform = `translate(${img._panX}px, ${img._panY}px) scale(${img._zoom})`;
}
