async function processImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    const processBtn = document.getElementById('processBtn');
    
    if (!file) {
        alert("Please select an image first.");
        return;
    }

    // Prevent multiple clicks
    processBtn.disabled = true;
    processBtn.style.opacity = 0.5;

    // Reset UI
    document.getElementById('statusBox').className = 'status-box waiting';
    document.getElementById('statusBox').innerText = 'Processing...';
    document.getElementById('reportText').innerText = '';
    document.getElementById('metricsPre').innerText = '';
    document.getElementById('overlayImg').src = '';
    document.getElementById('pathImg').src = '';
    
    // Display Original
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('originalImg').src = e.target.result;
    }
    reader.readAsDataURL(file);

    // Helper for fresh FormData
    const getFormData = () => {
        const fd = new FormData();
        fd.append('file', file);
        return fd;
    };

    try {
        // ---------------------------------------------------------
        // 1. Call /predict
        // Expects: { "overlay_image": "base64", "metrics": { ... } }
        // ---------------------------------------------------------
        console.log("Calling /predict...");
        const predictResponse = await fetch('/predict', {
            method: 'POST',
            body: getFormData()
        });
        
        if (!predictResponse.ok) throw new Error("/predict failed");
        const predictData = await predictResponse.json();
        console.log("Predict response:", predictData);

        // Bind Overlay
        if (predictData.overlay_image) {
            document.getElementById('overlayImg').src = "data:image/png;base64," + predictData.overlay_image;
        } else {
            console.error("Missing 'overlay_image' in /predict response");
            document.getElementById('statusBox').innerText = 'Error';
            document.getElementById('statusBox').className = 'status-box BLOCKED';
            processBtn.disabled = false;
            processBtn.style.opacity = 1;
            return;
        }

        // Bind Metrics (from /predict now)
        if (predictData.metrics) {
            document.getElementById('metricsPre').innerText = JSON.stringify(predictData.metrics, null, 2);
        } else {
            document.getElementById('metricsPre').innerText = "No metrics available";
            console.error("Missing 'metrics' in /predict response");
        }

        // ---------------------------------------------------------
        // 2. Call /path
        // Expects: { "path_image": "base64" }
        // ---------------------------------------------------------
        console.log("Calling /path...");
        const pathResponse = await fetch('/path', {
            method: 'POST',
            body: getFormData()
        });

        if (!pathResponse.ok) throw new Error("/path failed");
        const pathData = await pathResponse.json();
        console.log("Path response:", pathData);

        // Bind Path
        if (pathData.path_image) {
            document.getElementById('pathImg').src = "data:image/png;base64," + pathData.path_image;
        } else {
            console.error("Missing 'path_image' in /path response");
            document.getElementById('statusBox').innerText = 'Error';
            document.getElementById('statusBox').className = 'status-box BLOCKED';
            processBtn.disabled = false;
            processBtn.style.opacity = 1;
            return;
        }

        // ---------------------------------------------------------
        // 3. Call /report
        // Expects: { "status": "...", "summary": "..." }
        // ---------------------------------------------------------
        console.log("Calling /report...");
        const reportResponse = await fetch('/report', {
            method: 'POST',
            body: getFormData()
        });

        if (!reportResponse.ok) throw new Error("/report failed");
        const reportData = await reportResponse.json();
        console.log("Report response:", reportData);

        // Bind Status
        const status = reportData.status || "N/A";
        const statusBox = document.getElementById('statusBox');
        statusBox.innerText = status;
        statusBox.className = `status-box ${status}`;

        // Bind Summary
        const summary = reportData.summary || "No summary available";
        document.getElementById('reportText').innerText = summary;

        // Done: re-enable button
        processBtn.disabled = false;
        processBtn.style.opacity = 1;

    } catch (error) {
        console.error("Error during processing:", error);
        alert("An error occurred. Check console for details.");
        document.getElementById('statusBox').innerText = 'Error';
        document.getElementById('statusBox').className = 'status-box BLOCKED';
        processBtn.disabled = false;
        processBtn.style.opacity = 1;
    }
}
