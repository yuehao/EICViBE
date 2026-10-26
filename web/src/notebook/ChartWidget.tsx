// The viewer's chart in a notebook (ADR 0016): lines and points of the kernel's
// data, with the page's look and its hover readout of the nearest point.

import { Chart, type Marker, type Series } from "../components/Chart";
import type { WidgetModel } from "./kernel";
import { useNotebookScheme, useTrait } from "./scheme";

export function ChartWidget({ model }: { model: WidgetModel }) {
  const scheme = useNotebookScheme();
  const title = useTrait<string>(model, "title");
  const xLabel = useTrait<string>(model, "x_label");
  const yLabel = useTrait<string>(model, "y_label");
  const series = useTrait<Series[]>(model, "series");
  const markers = useTrait<Marker[]>(model, "markers");
  const log = useTrait<[number, number] | null>(model, "log");
  const xDomain = useTrait<[number, number] | null>(model, "x_domain");
  const yZero = useTrait<boolean>(model, "y_zero");
  const names = useTrait<string[]>(model, "names");
  const height = useTrait<number>(model, "height");
  return (
    <div
      className="eicvibe-widget nb-chart"
      data-scheme={scheme}
      style={{ ["--chart-height" as string]: `${height}px` }}
      data-testid="eicvibe-chart"
    >
      <Chart
        title={title}
        xLabel={xLabel}
        yLabel={yLabel}
        series={series}
        markers={markers}
        log={log ?? undefined}
        xDomain={xDomain ?? undefined}
        yZero={yZero}
        names={names.length ? names : undefined}
        scheme={scheme}
      />
    </div>
  );
}
