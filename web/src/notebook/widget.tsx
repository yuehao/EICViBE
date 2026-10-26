// The notebook widgets' module (ADR 0016), as anywidget loads it in JupyterLab,
// Notebook 7, VS Code and Colab: it draws one view into the element it is given
// (the kind the widget names: floor, optics, orbit or chart), with the page's
// components and stylesheet (scoped to the widget when built,
// vite.widget.config.ts), and asks the kernel for the view's data. The views are
// separate widgets in one module, so that React and the page's code come once.

import { StrictMode, type ComponentType } from "react";
import { createRoot } from "react-dom/client";

import type { ViewerData } from "../api/data";
import { installKindColors } from "../model/kinds";
import "../styles.css";
import "./widget.css";
import { ChartWidget } from "./ChartWidget";
import { FloorWidget } from "./FloorWidget";
import { kernelData, type WidgetModel } from "./kernel";
import { OpticsWidget } from "./OpticsWidget";
import { OrbitWidget } from "./OrbitWidget";

const VIEWS: Record<string, ComponentType<{ model: WidgetModel; data: ViewerData }>> = {
  floor: FloorWidget,
  optics: OpticsWidget,
  orbit: OrbitWidget,
  chart: ChartWidget,
};

function render({ model, el }: { model: WidgetModel; el: HTMLElement }) {
  const View = VIEWS[model.get("view") as string];
  if (!View) {
    el.textContent = `EICViBE: no view ${String(model.get("view"))}; rebuild the widgets (npm run build in web/)`;
    return;
  }
  installKindColors();
  const data = kernelData(model);
  const root = createRoot(el);
  root.render(
    <StrictMode>
      <View model={model} data={data} />
    </StrictMode>,
  );
  return () => {
    root.unmount();
    data.close();
  };
}

export default { render };
