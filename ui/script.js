async function sendQuery() {
  const query = document.getElementById('query').value;
  const res = await fetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });
  const data = await res.json();
  document.getElementById('response').textContent = JSON.stringify(data, null, 2);
}

document.getElementById('sendQuery').addEventListener('click', sendQuery);

async function uploadFile() {
  const fileInput = document.getElementById('fileInput');
  const formData = new FormData();
  formData.append('file', fileInput.files[0]);
  const res = await fetch('/upload', { method: 'POST', body: formData });
  const data = await res.json();
  document.getElementById('uploadStatus').textContent = data.status;
}

document.getElementById('uploadBtn').addEventListener('click', uploadFile);

async function refreshStatus() {
  const res = await fetch('/status');
  const data = await res.json();
  document.getElementById('statusData').textContent = JSON.stringify(data, null, 2);
}

document.getElementById('refreshStatus').addEventListener('click', refreshStatus);

async function refreshBayes() {
  const res = await fetch('/bayes');
  const data = await res.json();
  document.getElementById('bayesData').textContent = JSON.stringify(data, null, 2);
}

document.getElementById('refreshBayes').addEventListener('click', refreshBayes);

async function refreshLogs() {
  const res = await fetch('/logs');
  const data = await res.json();
  document.getElementById('logData').textContent = JSON.stringify(data, null, 2);
}

document.getElementById('refreshLogs').addEventListener('click', refreshLogs);
