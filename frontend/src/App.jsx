import React from "react";
import ChatPage from "./pages/ChatPage";
import ThemeProvider from "./providers/ThemeContext";
import "./styles/global.css";

export default function App() {
  return (
    <ThemeProvider>
      <ChatPage />
    </ThemeProvider>
  );
}