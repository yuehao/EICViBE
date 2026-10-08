import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { installKindColors } from "./model/kinds";
import "./styles.css";

installKindColors();

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
