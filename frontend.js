function analyzeVideo() {
    var videoUrl = document.getElementById('video_url').value;
    
    fetch('/custom_analyze', {  // Updated route name
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: videoUrl })
    })
    .then(response => response.json())
    .then(data => {
        var resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block'; // Show the result section
        var sentimentScoreElement = document.getElementById('sentiment_score');
        sentimentScoreElement.innerText = `Sentiment Score: ${data.sentiment_score}`;
    })
    .catch(error => console.error('Error:', error));
}
