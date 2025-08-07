import React from "react";
import "../styles/ChatBubble.css";

export default function ChatBubble({ role, content, drawio_link }) {
  const isAssistant = role === "assistant";
  const name = isAssistant ? "DrawioGPT" : "You";
  const avatar = isAssistant ? "/logo.svg" : "/profile-avatar.png";

  
  const xmlMatch = content.match(/(<mxfile>[\s\S]*<\/mxfile>)/);
  const xmlBlock = xmlMatch ? xmlMatch[1] : null;
  const rest = xmlBlock ? content.replace(xmlBlock, "") : content;

  return (
    <div className={`bubble-wrapper ${isAssistant ? "assistant" : "user"}`}>
      <div className={`bubble-label-row ${isAssistant ? "assistant" : "user"}`}>
        <img src={avatar} alt={name} className="bubble-avatar" />
        <span className="bubble-label">{name}</span>
      </div>
      <div className={`chat-bubble ${isAssistant ? "ai" : "user"}`}>
        {rest && <p>{rest.trim()}</p>}
        {xmlBlock && (
          <pre className="code-block">{xmlBlock}</pre>
        )}
        {isAssistant && drawio_link && (
          <div className="diagram-link-row">
            <a
              href={drawio_link}
              target="_blank"
              rel="noopener noreferrer"
              className="diagram-link"
            >
              View in Draw.io
            </a>
          </div>
        )}
      </div>
    </div>
  );
}