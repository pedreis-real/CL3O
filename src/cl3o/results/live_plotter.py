'''
================================================================================
CL3O - Composite Lifting Surface Structural Sizing & Optimization.
Live Plotter Module.

Live viewer for the Differential Evolution loop: a Matplotlib convergence plot
and an optional VTK 3-D wing-geometry window. Separated from main.py so the
optimization orchestration (RunCLEO) stays independent of the visualization
stack (Matplotlib / VTK).

@ CL3O Authors - MIT License
================================================================================
'''

# ================ PyLib imports ================
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
try:
    import vtk                       # only the optional live 3-D viewer needs it
except ImportError:                  # headless env without OpenGL (e.g. CI)
    vtk = None

# ================ Module imports ================

# Utilities
from cl3o.utils import io_utils as io

# Optimization
from cl3o.optimization.fobjective import RuntimeData
from cl3o.optimization.de_data import HistoryData

if TYPE_CHECKING:
    from cl3o.main import StaticData


class LivePlotter:
    '''
    Live viewer for the DE optimization loop.

    Opens two windows refreshed once per generation:
      - Matplotlib window : DE convergence curve with std shaded band.
      - VTK window        : 3-D wing geometry of the current best individual
                            (outer skin, booms, spar surfaces, centroid and
                            shear-centre lines). Camera is fixed at the
                            top-aft-left corner (Xmin, Ymin, Zmax).
    '''

    def __init__(
        self,
        static_data    : StaticData,
        enable_logging : bool = True,
    ) -> None:
        '''
        Args:
            static_data   : StaticData (wing geometry reference).
            enable_logging: Toggle logger.
        '''
        if vtk is None:
            raise RuntimeError(
                "The live 3-D viewer requires VTK, which failed to import "
                "(no OpenGL/libGL in this environment). Run with "
                "live_plot=False or install OpenGL libraries (e.g. libgl1)."
            )
        self.logger = io.setup_logger(self, enable_logging)
        self.st     = static_data

        # -------- Convergence panel (matplotlib) --------
        plt.ion()
        self.fig     = plt.figure(figsize=(5, 7))
        self.ax_conv = self.fig.add_subplot(1, 1, 1)
        self.fig.tight_layout()
        plt.show(block=False)
        plt.pause(0.05)

        # Pin convergence window to the top-right corner of the screen
        _mgr = plt.get_current_fig_manager()
        try:
            import tkinter as _tk
            _root = _tk.Tk(); _root.withdraw()
            _sw   = _root.winfo_screenwidth()
            _root.destroy()
            _fw   = int(self.fig.get_size_inches()[0] * self.fig.dpi)
            _x    = max(0, _sw - _fw - 10)
            _mgr.window.update_idletasks()
            _mgr.window.wm_geometry(f"+{_x}+0")
            _mgr.window.update_idletasks()
        except Exception:
            try:
                _mgr.window.move(1290, 0)
            except Exception:
                pass

        # -------- Wing geometry panel (VTK) --------
        self._setup_vtk_window()

        self.logger.info("LivePlotter ready.")

    # ----------------------------------------
    # Public method - per-generation update
    # ----------------------------------------

    def update(
        self,
        k    : int,
        hist : HistoryData,
        rt   : RuntimeData,
    ) -> None:
        '''
        Redraw both panels for generation k.

        Args:
            k   : Current generation index.
            hist: HistoryData snapshot trimmed to [0, k].
            rt  : RuntimeData of the current best individual.
        '''
        self._update_convergence(k, hist)
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
        self._update_wing_vtk(rt)

    # ------------------------------------------------
    # Private method - VTK window initialisation
    # ------------------------------------------------

    def _setup_vtk_window(self) -> None:
        '''Initialise VTK renderer, render window, and interactor.'''
        self._ren = vtk.vtkRenderer()
        self._ren.SetBackground(0.12, 0.12, 0.12)

        self._rw = vtk.vtkRenderWindow()
        self._rw.AddRenderer(self._ren)
        self._rw.SetSize(1280, 720)
        self._rw.SetWindowName("CL3O - Wing Geometry")

        self._iren = vtk.vtkRenderWindowInteractor()
        self._iren.SetRenderWindow(self._rw)
        self._iren.SetInteractorStyle(
            vtk.vtkInteractorStyleTrackballCamera()
        )
        self._iren.Initialize()
        self._rw.Render()

    # ----------------------------------------
    # Private method - convergence subplot
    # ----------------------------------------

    def _update_convergence(
        self,
        k    : int,
        hist : HistoryData,
    ) -> None:
        '''Redraw convergence with best-f curve and population std band.'''
        ax = self.ax_conv
        ax.cla()

        gens = np.arange(hist.ng + 1)

        ax.plot(
            gens, hist.best_f,
            color="#085302", lw=1.5, label='best f',
        )
        ax.fill_between(
            gens,
            hist.mean_f - hist.std_f,
            hist.mean_f + hist.std_f,
            alpha=0.25,
            color='#4DB8FF',
            label=r'mean $\pm$ std',
        )
        if hist.feasible_f < float('inf'):
            ax.axhline(
                hist.feasible_f,
                color='green',
                linestyle='--',
                lw=1.2,
                label=f'Feasible: {hist.feasible_f:.3f}',
            )

        ax.set_xlabel(r'Generation', fontsize=10)
        ax.set_ylabel(r'Fitness z(X)', fontsize=10)
        ax.tick_params(axis='both', labelsize=6)

        ax.set_title(f'DE Convergence  [gen {k}]')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # ----------------------------------------
    # Private method - VTK wing geometry
    # ----------------------------------------

    def _update_wing_vtk(self, rt: RuntimeData) -> None:
        '''
        Rebuild all VTK actors for the current best-individual pipeline.

        Coordinate layout: X = chord direction, Y = span, Z = vertical.
        Boom global coords: X = C[0] + boom_u[j], Z = C[2] + boom_w[j].
        Boom order (0-indexed): B1=0 .. B7=6.
        '''
        ren = self._ren
        ren.RemoveAllViewProps()

        secs = rt.sections
        Y    = np.array([sd.C[1] for sd in secs.sec_data])
        bX   = np.array(
            [[sd.C[0] + sd.boom_u[j] for j in range(7)]
             for sd in secs.sec_data]
        )
        bZ   = np.array(
            [[sd.C[2] + sd.boom_w[j] for j in range(7)]
             for sd in secs.sec_data]
        )

        # -------- (a) Outer wing skin surface --------
        profiles = []
        for sd in secs.sec_data:
            outer = np.concatenate([
                sd.T1[0]['pts'],
                sd.T1[1]['pts'][1:],
                sd.T1[2]['pts'][1:],
                sd.T1[3]['pts'][1:],
                sd.T1[4]['pts'][1:],
            ], axis=0)
            profiles.append(outer)
        ren.AddActor(LivePlotter._skin_actor(profiles, Y))
        ren.AddActor(LivePlotter._section_outlines_actor(profiles, Y))

        _SPAR_EDGE = (0.28, 0.28, 0.28)

        # -------- (b) Aft spar: B3 (idx 2) to B5 (idx 4) --------
        ren.AddActor(LivePlotter._spar_actor(
            Y,
            bX[:, 2], bZ[:, 2], bX[:, 4], bZ[:, 4],
            color=(0.27, 0.51, 0.71), alpha=0.72,
            edge_color=_SPAR_EDGE,
        ))

        # -------- (c) Rear spar: B1 (idx 0) to B7 (idx 6) --------
        ren.AddActor(LivePlotter._spar_actor(
            Y,
            bX[:, 0], bZ[:, 0], bX[:, 6], bZ[:, 6],
            color=(1.00, 0.39, 0.28), alpha=0.72,
            edge_color=_SPAR_EDGE,
        ))

        # -------- Axes labels and title overlay --------
        all_X = bX.ravel()
        all_Z = bZ.ravel()
        bounds = (
            float(all_X.min()), float(all_X.max()),
            float(Y.min()),     float(Y.max()),
            float(all_Z.min()), float(all_Z.max()),
        )
        LivePlotter._add_axes(ren, bounds)
        ren.AddActor2D(
            LivePlotter._title_actor("Wing Geometry (best individual)")
        )

        # -------- Camera: top-aft-left (Xmin, Ymin, Zmax) --------
        self._set_camera(ren, all_X, Y, all_Z)

        self._rw.Render()
        try:
            self._iren.ProcessEvents()
        except AttributeError:
            pass

    # ----------------------------------------
    # Private method - camera positioning
    # ----------------------------------------

    def _set_camera(
        self,
        ren   : object,
        all_X : np.ndarray,
        all_Y     : np.ndarray,
        all_Z : np.ndarray,
    ) -> None:
        '''
        Position camera at the top-aft-left corner (Xmin, Ymin, Zmax).

        Args:
            ren  : vtkRenderer instance.
            all_X: All X-coordinate values for bounds computation.
            Y    : Spanwise Y-coordinate array.
            all_Z: All Z-coordinate values for bounds computation.
        '''
        cx = float((all_X.min() + all_X.max()) * 0.60)
        cy = float((all_Y.min() + all_Y.max()) * 0.60)
        cz = float((all_Z.min() + all_Z.max()) * 0.60)
        span = float(max(
            all_X.max() - all_X.min(),
            all_Y.max() - all_Y.min(),
            all_Z.max() - all_Z.min(),
        ))
        cam = ren.GetActiveCamera()
        cam.SetFocalPoint(cx, cy, cz)
        cam.SetPosition(
            float(all_X.min()) - 0.50 * span,
            float(all_Y.min()) - 0.30 * span,
            float(all_Z.max()) + 0.35 * span,
        )
        cam.SetViewUp(0.0, 0.0, 1.0)
        ren.ResetCameraClippingRange()

    # ------------------------------------------------
    # Private static methods - VTK actor builders
    # ------------------------------------------------

    @staticmethod
    def _skin_actor(
        profiles : list,
        Y        : np.ndarray,
    ) -> object:
        '''
        Build a transparent gray surface from adjacent profile quads.

        Args:
            profiles: List of (N, 2) outer-skin point arrays per station.
            Y       : Spanwise Y coordinates.

        Returns:
            vtkActor for the outer skin surface.
        '''
        pts   = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        pid   = 0
        for j in range(len(Y) - 1):
            p0, p1 = profiles[j], profiles[j + 1]
            y0, y1 = float(Y[j]), float(Y[j + 1])
            m = min(len(p0), len(p1)) - 1
            for i in range(m):
                x0, z0 = float(p0[i,   0]), float(p0[i,   1])
                x1, z1 = float(p0[i+1, 0]), float(p0[i+1, 1])
                x2, z2 = float(p1[i+1, 0]), float(p1[i+1, 1])
                x3, z3 = float(p1[i,   0]), float(p1[i,   1])
                pts.InsertNextPoint(x0, y0, z0)
                pts.InsertNextPoint(x1, y0, z1)
                pts.InsertNextPoint(x2, y1, z2)
                pts.InsertNextPoint(x3, y1, z3)
                quad = vtk.vtkQuad()
                quad.GetPointIds().SetId(0, pid)
                quad.GetPointIds().SetId(1, pid + 1)
                quad.GetPointIds().SetId(2, pid + 2)
                quad.GetPointIds().SetId(3, pid + 3)
                cells.InsertNextCell(quad)
                pid += 4

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.88, 0.88, 0.88)
        actor.GetProperty().SetOpacity(0.28)
        actor.GetProperty().SetEdgeVisibility(0)
        return actor

    @staticmethod
    def _section_outlines_actor(
        profiles : list,
        Y        : np.ndarray,
    ) -> object:
        '''
        Build a single actor with closed cross-section outline polylines
        at every spanwise station.

        Args:
            profiles: List of (N, 2) outer-skin point arrays per station.
            Y       : Spanwise Y coordinates (one per station).

        Returns:
            vtkActor containing one closed polyline per station.
        '''
        pts    = vtk.vtkPoints()
        cells  = vtk.vtkCellArray()
        offset = 0
        for prof, y in zip(profiles, Y):
            n = len(prof)
            yf = float(y)
            for x, z in prof:
                pts.InsertNextPoint(float(x), yf, float(z))
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(n + 1)
            for i in range(n):
                pl.GetPointIds().SetId(i, offset + i)
            pl.GetPointIds().SetId(n, offset)
            cells.InsertNextCell(pl)
            offset += n

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.92, 0.92, 0.92)
        actor.GetProperty().SetLineWidth(1.0)
        actor.GetProperty().SetOpacity(0.70)
        return actor

    @staticmethod
    def _spar_actor(
        Y          : np.ndarray,
        X1         : np.ndarray,
        Z1         : np.ndarray,
        X2         : np.ndarray,
        Z2         : np.ndarray,
        color      : tuple,
        alpha      : float,
        edge_color : tuple | None = None,
    ) -> object:
        '''
        Build a quad-strip actor between two boom lines.

        Args:
            Y         : Spanwise Y coordinates.
            X1/Z1     : Chord/vertical coords of the first boom line.
            X2/Z2     : Chord/vertical coords of the second boom line.
            color     : (r, g, b) float triple in [0, 1].
            alpha     : Surface opacity.
            edge_color: When supplied, draw quad edges with this color.

        Returns:
            vtkActor for the spar web surface.
        '''
        pts   = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
        pid   = 0
        for j in range(len(Y) - 1):
            p0x, p0z = float(X1[j]),   float(Z1[j])
            p1x, p1z = float(X1[j+1]), float(Z1[j+1])
            q0x, q0z = float(X2[j]),   float(Z2[j])
            q1x, q1z = float(X2[j+1]), float(Z2[j+1])
            y0,  y1  = float(Y[j]),    float(Y[j+1])
            pts.InsertNextPoint(p0x, y0, p0z)
            pts.InsertNextPoint(p1x, y1, p1z)
            pts.InsertNextPoint(q1x, y1, q1z)
            pts.InsertNextPoint(q0x, y0, q0z)
            quad = vtk.vtkQuad()
            for i in range(4):
                quad.GetPointIds().SetId(i, pid + i)
            cells.InsertNextCell(quad)
            pid += 4

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(alpha)
        if edge_color is not None:
            actor.GetProperty().SetEdgeVisibility(1)
            actor.GetProperty().SetEdgeColor(*edge_color)
        else:
            actor.GetProperty().SetEdgeVisibility(0)
        return actor

    @staticmethod
    def _line_actor(
        X     : np.ndarray,
        Y     : np.ndarray,
        Z     : np.ndarray,
        color : tuple,
        lw    : float = 1.5,
    ) -> object:
        '''
        Build a spanwise polyline actor.

        Args:
            X/Y/Z: Coordinate arrays along the line.
            color: (r, g, b) float triple in [0, 1].
            lw   : Line width in points.

        Returns:
            vtkActor for the polyline.
        '''
        pts = vtk.vtkPoints()
        for x, y, z in zip(X, Y, Z):
            pts.InsertNextPoint(float(x), float(y), float(z))

        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(len(X))
        for i in range(len(X)):
            pl.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(pl)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetLineWidth(lw)
        return actor

    @staticmethod
    def _add_axes(ren: object, bounds: tuple) -> None:
        '''
        Attach a vtkCubeAxesActor labelled X/Y/Z [mm] to the renderer.

        Args:
            ren   : vtkRenderer.
            bounds: (xmin, xmax, ymin, ymax, zmin, zmax) bounding box.
        '''
        axes = vtk.vtkCubeAxesActor()
        axes.SetBounds(*bounds)
        axes.SetCamera(ren.GetActiveCamera())
        axes.SetXTitle("X [mm]")
        axes.SetYTitle("Y [mm]")
        axes.SetZTitle("Z [mm]")
        for i in range(3):
            axes.GetTitleTextProperty(i).SetColor(1.0, 1.0, 1.0)
            axes.GetLabelTextProperty(i).SetColor(0.8, 0.8, 0.8)
        axes.GetXAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.GetYAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.GetZAxesLinesProperty().SetColor(1.0, 1.0, 1.0)
        axes.SetFlyModeToOuterEdges()
        ren.AddActor(axes)

    @staticmethod
    def _title_actor(text: str) -> object:
        '''Return a 2D screen-space text actor at the bottom-left corner.'''
        ta = vtk.vtkTextActor()
        ta.SetInput(text)
        ta.GetTextProperty().SetFontSize(13)
        ta.GetTextProperty().SetColor(1.0, 1.0, 1.0)
        ta.SetPosition(10, 10)
        return ta
