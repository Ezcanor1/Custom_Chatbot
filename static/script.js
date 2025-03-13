function sendMessage() {
    let userInput = document.getElementById("user-input").value.trim();
    let chatBox = document.getElementById("chat-box");

    if (!userInput) return;

    // Append user message
    let userMessage = document.createElement("div");
    userMessage.innerHTML = `<strong>You:</strong> ${userInput}`;
    chatBox.appendChild(userMessage);

    // Append AI thinking message
    let botMessage = document.createElement("div");
    botMessage.innerHTML = `<strong>Bot:</strong> AI is thinking...`;
    chatBox.appendChild(botMessage);

    // Send request to backend using EventSource for streaming
    fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userInput })
    })
    .then(response => {
        const reader = response.body.getReader();
        let decoder = new TextDecoder();
        let responseText = "";

        function readChunk() {
            return reader.read().then(({ done, value }) => {
                if (done) {
                    return;
                }
                let chunk = decoder.decode(value, { stream: true });
                
                try {
                    let jsonData = JSON.parse(chunk);
                    if (jsonData.typing) {
                        botMessage.innerHTML = `<strong>Bot:</strong> ${jsonData.message}`;
                    } else {
                        responseText += jsonData.message + " ";
                        botMessage.innerHTML = `<strong>Bot:</strong> ${responseText}`;
                    }
                } catch (error) {
                    console.error("JSON Parse Error:", error);
                }

                return readChunk();
            });
        }

        return readChunk();
    })
    .catch(error => {
        botMessage.innerHTML = `<strong>Error:</strong> ${error.message}`;
    });

    document.getElementById("user-input").value = "";
}
