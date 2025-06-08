const form = document.getElementById("upload-form");
const statusDiv = document.getElementById("status");
const resultVideo = document.getElementById("result-video");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const input = document.getElementById("video-input");
  if (!input.files.length) return;

  const file = input.files[0];
  const fd = new FormData();
  fd.append("video", file);

  statusDiv.textContent = "Uploading and analyzing…";

  try {
    // 1) Upload & trigger processing
    const uploadResp = await fetch("/v1/analyze/upload", {
      method: "POST",
      body: fd,
    });
    if (!uploadResp.ok) {
      throw new Error(`Upload failed: ${uploadResp.statusText}`);
    }
    const uploadJson = await uploadResp.json();
    statusDiv.textContent = uploadJson.status || "Analysis complete.";

    // 2) Fetch the processed video as a blob
    const videoResp = await fetch("/v1/analyze/result");
    if (!videoResp.ok) {
      throw new Error(`Could not fetch result video: ${videoResp.statusText}`);
    }
    const blob = await videoResp.blob();

    // 3) Create an object URL and assign to the video element
    const url = URL.createObjectURL(blob);
    resultVideo.src = url;
    resultVideo.style.display = "block";
    resultVideo.load();
    resultVideo.play();
  } catch (err) {
    console.error(err);
    statusDiv.textContent = err.message;
  }
});
