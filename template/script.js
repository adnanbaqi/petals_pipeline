document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file-input');
    const progressBarFill = document.querySelector('#progress-bar-fill');
    const progressBarContainer = document.getElementById('progress-bar-container');
    const uploadSuccess = document.querySelector('#upload-success');
    const proceedButton = document.querySelector('.proceed-button');
    const languageDisplay = document.getElementById('language-display');
    const tokensDisplay = document.getElementById('tokens-display');
    let selectedFile = null; // Variable to store the selected file

    // Elements to display response data

    if (!fileInput || !progressBarFill || !uploadSuccess || !proceedButton || !languageDisplay || !tokensDisplay) {
        console.error('One or more elements are missing in the HTML document.');
        return;
    }

    fileInput.addEventListener('change', function(e) {
        selectedFile = e.target.files[0]; // Store the selected file
    });

    proceedButton.addEventListener('click', function() {
        if (selectedFile !== null) {
            uploadFile(selectedFile); // Upload the stored file
        } else {
            console.log('No file selected.');
            // Optionally, you can re-open the file dialog here if you want the user to select a file
            // fileInput.click();
        }
    });

    function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
    
        fetch('http://127.0.0.1:8000/api/v1/store', {
            method: 'POST',
            body: formData,
        }).then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        }).then(data => {
            console.log('Upload successful:', data);
            alert('File has been uploaded and processed successfully.');
    
            // Update the webpage with the response data
            languageDisplay.textContent = data.language;
            tokensDisplay.textContent = data.num_tokens;
    
            // Hide progress bar after upload is complete
            progressBarContainer.style.display = 'none';
            uploadSuccess.style.display = 'block'; // Show upload success message
    
        }).catch(error => {
            console.error('Upload error:', error);
        });
    
        // This part simulates the progress update. Replace this with real progress updates if your API supports it.
        let progress = 0;
        const simulateUpload = setInterval(() => {
            progress += 10;
            progressBarFill.style.width = progress + '%';
            if (progress >= 100) clearInterval(simulateUpload);
        }, 200); // Adjust time to simulate upload speed
    }
    
});
