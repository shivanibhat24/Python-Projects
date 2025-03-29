// Updated frontend JavaScript code to integrate with the backend

// Global variables
let selectedJobId = null;
let statusCheckInterval = null;

// File Upload and Processing
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewImage = document.getElementById('preview-image');
const transformButton = document.getElementById('transform-button');
const animateCheckbox = document.getElementById('animate-checkbox');
const animationOptions = document.getElementById('animation-options');
const styleOptions = document.querySelectorAll('.style-option');
const resultsContainer = document.getElementById('results');
const loadingIndicator = document.getElementById('loading');

// Show/hide animation options
animateCheckbox.addEventListener('change', () => {
    animationOptions.style.display = animateCheckbox.checked ? 'block' : 'none';
});

// Handle file selection
fileInput.addEventListener('change', handleFileSelect);

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file && file.type.match('image.*')) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
            transformButton.disabled = false;
        }
        
        reader.readAsDataURL(file);
    }
}

// Style selection
styleOptions.forEach(option => {
    option.addEventListener('click', () => {
        // Deselect all
        styleOptions.forEach(opt => opt.classList.remove('selected'));
        // Select clicked
        option.classList.add('selected');
    });
});

// Handle drag and drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.backgroundColor = 'rgba(126, 90, 255, 0.1)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.backgroundColor = '';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.backgroundColor = '';
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type.match('image.*')) {
        fileInput.files = files;
        handleFileSelect({ target: { files: files } });
    }
});

// Handle transform button click
transformButton.addEventListener('click', async () => {
    if (!fileInput.files || fileInput.files.length === 0) {
        alert('Please select an image first');
        return;
    }
    
    // Get selected style
    let selectedStyle = 'ghibli'; // Default
    styleOptions.forEach(option => {
        if (option.classList.contains('selected')) {
            selectedStyle = option.dataset.style.toLowerCase();
        }
    });
    
    // Animation options
    const animated = animateCheckbox.checked;
    const frames = document.getElementById('frames-input').value;
    const transition = document.getElementById('transition-checkbox').checked;
    
    // Show loading
    loadingIndicator.style.display = 'block';
    resultsContainer.innerHTML = '';
    transformButton.disabled = true;
    
    try {
        // Create form data for file upload
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        formData.append('style', selectedStyle);
        formData.append('animated', animated.toString());
        formData.append('frames', frames);
        formData.append('transition', transition.toString());
        
        // Upload file and start processing
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Error uploading image');
        }
        
        const data = await response.json();
        selectedJobId = data.job_id;
        
        // Start checking status periodically
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
        
        statusCheckInterval = setInterval(checkJobStatus, 2000);
        
    } catch (error) {
        // Handle error
        loadingIndicator.style.display = 'none';
        transformButton.disabled = false;
        alert('Error: ' + error.message);
        console.error('Upload error:', error);
    }
});

// Check job status
async function checkJobStatus() {
    if (!selectedJobId) return;
    
    try {
        const response = await fetch(`/api/status/${selectedJobId}`);
        
        if (!response.ok) {
            throw new Error('Failed to get job status');
        }
        
        const data = await response.json();
        
        // Update loading indicator with progress
        if (data.status === 'processing') {
            const loadingText = document.querySelector('#loading p');
            loadingText.textContent = `Processing your image... ${data.progress}%`;
        }
        
        // If completed, show results
        if (data.status === 'completed') {
            clearInterval(statusCheckInterval);
            loadingIndicator.style.display = 'none';
            transformButton.disabled = false;
            displayResults(data.result_files);
        }
        
        // If failed, show error
        if (data.status === 'failed') {
            clearInterval(statusCheckInterval);
            loadingIndicator.style.display = 'none';
            transformButton.disabled = false;
            alert('Processing failed: ' + (data.error || 'Unknown error'));
        }
        
    } catch (error) {
        console.error('Status check error:', error);
    }
}

// Display results
function displayResults(resultFiles) {
    resultsContainer.innerHTML = '';
    
    // Get style name for display
    let styleName = 'Default';
    styleOptions.forEach(option => {
        if (option.classList.contains('selected')) {
            styleName = option.dataset.style;
        }
    });
    
    // Check if we have animation files (gif or mp4)
    const animationFiles = resultFiles.filter(file => 
        file.endsWith('.gif') || file.endsWith('.mp4')
    );
    
    if (animationFiles.length > 0) {
        // Create result card for animation
        const animationCard = document.createElement('div');
        animationCard.className = 'result-card';
        
        // Use the GIF for preview
        const gifFile = animationFiles.find(file => file.endsWith('.gif'));
        
        animationCard.innerHTML = `
            <img src="/api/results/${gifFile}" class="result-image">
            <div class="result-info">
                <h4>Anime Animation</h4>
                <p>Style: ${styleName}</p>
                ${animateCheckbox.checked && document.getElementById('transition-checkbox').checked ? 
                    '<p>Style transition enabled</p>' : ''}
                <p>Frames: ${document.getElementById('frames-input').value}</p>
                ${animationFiles.map(file => 
                    `<a href="/api/results/${file}" download class="download-button">
                        Download ${file.endsWith('.gif') ? 'GIF' : 'MP4'}
                    </a>`
                ).join(' ')}
            </div>
        `;
        
        resultsContainer.appendChild(animationCard);
        
        // Also display selected frames
        const frameFiles = resultFiles.filter(file => 
            file.includes('_frame_') && file.endsWith('.png')
        );
        
        // Display up to 4 frames
        const framesToShow = frameFiles.slice(0, 4);
        
        if (framesToShow.length > 0) {
            const framesContainer = document.createElement('div');
            framesContainer.style.marginTop = '20px';
            framesContainer.innerHTML = '<h3>Selected Frames</h3>';
            
            const framesGrid = document.createElement('div');
            framesGrid.className = 'results-grid';
            
            framesToShow.forEach(frameFile => {
                const frameCard = document.createElement('div');
                frameCard.className = 'result-card';
                frameCard.innerHTML = `
                    <img src="/api/results/${frameFile}" class="result-image">
                    <div class="result-info">
                        <p>Frame ${frameFile.match(/_frame_(\d+)/)[1]}</p>
                        <a href="/api/results/${frameFile}" download class="download-button">
                            Download
                        </a>
                    </div>
                `;
                framesGrid.appendChild(frameCard);
            });
            
            framesContainer.appendChild(framesGrid);
            resultsContainer.appendChild(framesContainer);
        }
        
    } else {
        // For single image result
        const imageFiles = resultFiles.filter(file => file.endsWith('.png') || file.endsWith('.jpg'));
        
        if (imageFiles.length > 0) {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';
            resultCard.innerHTML = `
                <img src="/api/results/${imageFiles[0]}" class="result-image">
                <div class="result-info">
                    <h4>Anime Style Transfer</h4>
                    <p>Style: ${styleName}</p>
                    <a href="/api/results/${imageFiles[0]}" download class="download-button">
                        Download Image
                    </a>
                </div>
            `;
            resultsContainer.appendChild(resultCard);
        }
    }
}

// Load gallery images from API
async function loadGallery() {
    try {
        const response = await fetch('/api/gallery');
        if (!response.ok) {
            throw new Error('Failed to load gallery');
        }
        
        const galleryItems = await response.json();
        
        // Find gallery container
        const galleryGrid = document.querySelector('#gallery .results-grid');
        if (!galleryGrid) return;
        
        // Clear existing items
        galleryGrid.innerHTML = '';
        
        // Add items from API
        galleryItems.forEach(item => {
            const galleryCard = document.createElement('div');
            galleryCard.className = 'result-card';
            galleryCard.innerHTML = `
                <img src="${item.image_url}" class="result-image" alt="${item.title}">
                <div class="result-info">
                    <h4>${item.title}</h4>
                    <p>By ${item.username}</p>
                    <p>${item.likes} likes</p>
                </div>
            `;
            galleryGrid.appendChild(galleryCard);
        });
        
    } catch (error) {
        console.error('Error loading gallery:', error);
    }
}

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Load gallery from API
    setTimeout(loadGallery, 1000);  // Slight delay to allow the gallery section to be created
    
    // Create contact form submit handler
    const contactForm = document.getElementById('contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                name: document.getElementById('name').value,
                email: document.getElementById('email').value,
                message: document.getElementById('message').value
            };
            
            try {
                // In a real implementation, you would send this to an API endpoint
                // For demo purposes, we'll just show success
                alert('Thank you for your message! We will get back to you soon.');
                this.reset();
            } catch (error) {
                alert('Error sending message. Please try again.');
                console.error('Form submission error:', error);
            }
        });
    }
});
