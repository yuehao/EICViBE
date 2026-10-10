import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "../styles.css";
import { CheckPage } from "./CheckPage";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <CheckPage />
  </StrictMode>,
);
