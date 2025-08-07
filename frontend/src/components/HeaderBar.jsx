import React from "react";
import { useTheme } from "../providers/ThemeContext";
import "../styles/HeaderBar.css";

export default function HeaderBar() {
  const { toggle, theme } = useTheme();

  return (
    <header className="header">
      <div className="logo">
        <img src="/logo.svg" alt="Logo" className="logo-img" />
        DrawioGPT
      </div>
      <div className="header-actions">
        <button className="theme-toggle" onClick={toggle}>
          <img
            src={theme === "dark" ? "/brightness.png" : "/moon.png"}
            alt={theme === "dark" ? "Light mode" : "Dark mode"}
            className="theme-icon"
          />
        </button>
        <img
          src="/profile-avatar.png"
          alt="Profile"
          className="profile-avatar"
        />
      </div>
    </header>
  );
}