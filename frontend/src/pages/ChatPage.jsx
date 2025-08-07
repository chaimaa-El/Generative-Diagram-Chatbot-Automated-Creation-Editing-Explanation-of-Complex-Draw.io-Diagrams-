import React, { useState } from "react";
import HeaderBar from "../components/HeaderBar";
import ChatBubble from "../components/ChatBubble";
import ChatInputBar from "../components/ChatInputBar";
import { sendMessage } from "../utils/api";

// Intent detection for explain
function isExplainIntent(text) {
  return /explain|describe|analyze|what does|summarize|interpret/i.test(text);
}

export default function ChatPage() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSend = async (text) => {
    if (!text.trim()) return;
    setLoading(true);
    setMessages([...messages, { role: "user", content: text }]);
    try {
      // Friendly assistant intro
      let intro = "";
      if (isExplainIntent(text)) {
        intro = "Here is an explanation of your diagram:\n\n";
      } else if (/diagram|uml|cnn|network|class|flow|use case|architecture/i.test(text)) {
        intro = "Sure! Here is an example of a diagram based on your request:\n\n";
      } else {
        intro = "Sure! Here is the result:\n\n";
      }
      const reply = await sendMessage([{ role: "user", content: text }]);
      setMessages([
        ...messages,
        { role: "user", content: text },
        {
          role: "assistant",
          content: intro + (reply.content || ""),
          drawio_link: reply.drawio_link || null,
        },
      ]);
    } catch (err) {
      console.error(err);
      setMessages([
        ...messages,
        { role: "user", content: text },
        {
          role: "assistant",
          content: "Sorry, something went wrong 💔",
        },
      ]);
    }
    setLoading(false);
  };

  return (
    <div className="chat-page">
      <HeaderBar />
      <div className="chat-window">
        {messages.map((msg, idx) => (
          <ChatBubble key={idx} {...msg} />
        ))}
        {loading && <ChatBubble role="assistant" content="..." />}
      </div>
      <ChatInputBar onSend={handleSend} loading={loading} />
    </div>
  );
}