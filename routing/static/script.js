document.addEventListener('DOMContentLoaded', () => {
    const chatLog = document.getElementById('chat-log');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    // Function to add a message to the chat log
    // Returns the message bubble element for potential later updates
    function addMessage(text, sender, isLoading = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', sender);

        const bubbleElement = document.createElement('div');
        bubbleElement.classList.add('message-bubble');

        if (isLoading) {
            bubbleElement.classList.add('loading-indicator');
            bubbleElement.textContent = text;
        } else {
             bubbleElement.innerHTML = text.replace(/\n/g, '<br>');
        }


        messageElement.appendChild(bubbleElement);
        chatLog.appendChild(messageElement);

        // Auto-scroll 
        chatLog.scrollTop = chatLog.scrollHeight;

        return bubbleElement;
    }

     // Function to update a loading message with final content and metrics
    function updateMessage(bubbleElement, text, metrics = null) {
        bubbleElement.classList.remove('loading-indicator');
        bubbleElement.innerHTML = text.replace(/\n/g, '<br>'); 

        if (metrics) {
            const metricsElement = document.createElement('div');
            metricsElement.classList.add('message-metrics');
            metricsElement.textContent = `Cluster: ${metrics.cluster}, Model: ${metrics.model_type}, ${metrics.generated_tokens} tokens, ${metrics.total_time.toFixed(3)}s (${metrics.tokens_per_second.toFixed(2)} tok/s)`;
            bubbleElement.appendChild(metricsElement);
        }

        // Auto-scroll 
        chatLog.scrollTop = chatLog.scrollHeight;
    }


    // Function to send message to backend
    async function sendMessage() {
        const prompt = userInput.value.trim();
        if (!prompt) return; 

        addMessage(prompt, 'user');
        userInput.value = ''; 

        const assistantBubble = addMessage("Assistant is thinking...", 'assistant', true);


        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ prompt: prompt }),
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' })); 
                const errorMessage = errorData.error || response.statusText;
                updateMessage(assistantBubble, `Error: ${errorMessage}`); 
                console.error('Backend error:', response.status, response.statusText, errorData);
                return;
            }

            const data = await response.json();

            // Updating the placeholder message with the actual response and metrics
            updateMessage(assistantBubble, data.response, data.metrics);


        } catch (error) {
            console.error('Fetch error:', error);
            updateMessage(assistantBubble, `An error occurred: ${error.message}`);
        }
    }

    sendButton.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            sendMessage();
            event.preventDefault(); 
        }
    });

});