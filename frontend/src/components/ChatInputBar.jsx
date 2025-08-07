import React, { useState } from "react";
import "../styles/ChatInputBar.css";

export default function ChatInputBar({ onSend, loading }) {
  const [input, setInput] = useState("");

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) {
        onSend(input);
        setInput("");
      }
    }
  };

  const handleSend = () => {
    if (input.trim()) {
      onSend(input);
      setInput("");
    }
  };

  return (
    <div className="input-bar">
      <input
        className="text-input"
        disabled={loading}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask me to generate, explain, or edit a diagram…"
      />
      <input type="file" id="upload" hidden />
      <label htmlFor="upload" className="upload-btn" title="Attach XML">
        <img src="/clip.png" alt="Attach" className="clip-icon" />
      </label>
      <button className="send-btn" onClick={handleSend} disabled={loading}>Send</button>
    </div>
  );
}