// src/utils/api.js
import axios from "axios";
const BASE_URL = "http://localhost:8000";

export async function sendMessage(messages) {
  // messages should be an array of {role, content}
  const res = await axios.post(`${BASE_URL}/chat`, { messages });
  return res.data;
}