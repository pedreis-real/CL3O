import { useEffect } from "react";
import { useStore } from "./state/store";
import { RunPicker } from "./components/RunPicker";
import { ToggleBar } from "./components/ToggleBar";
import { Sidebar } from "./components/Sidebar";
import { GenerationSlider } from "./components/GenerationSlider";
import { GeometryPlot } from "./plots/GeometryPlot";
import { SectionPlot } from "./plots/SectionPlot";
import { MeshPlot } from "./plots/MeshPlot";
import { StressPlot } from "./plots/StressPlot";
import { MiscPlot } from "./plots/MiscPlot";

export default function App() {
  const { view, init } = useStore();

  useEffect(() => {
    void init();
  }, [init]);

  return (
    <div className="app">
      <header className="topbar">
        <RunPicker />
        <ToggleBar />
      </header>

      <main className="canvas">
        {view === "geometry" && <GeometryPlot />}
        {view === "section" && <SectionPlot />}
        {view === "mesh" && <MeshPlot />}
        {view === "stress" && <StressPlot />}
        {view === "misc" && <MiscPlot />}
      </main>

      <Sidebar />

      <footer className="bottombar">
        <GenerationSlider />
      </footer>
    </div>
  );
}
