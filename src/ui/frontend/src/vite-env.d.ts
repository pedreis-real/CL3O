/// <reference types="vite/client" />

// plotly.js-dist-min ships no types; react-plotly.js/factory is untyped too.
declare module "plotly.js-dist-min";
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  import type { PlotParams } from "react-plotly.js";
  const createPlotlyComponent: (plotly: unknown) => ComponentType<PlotParams>;
  export default createPlotlyComponent;
}
