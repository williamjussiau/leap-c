import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver

plt.ion()


class CylinderRenderer:
    def __init__(self, flowsolver: CylinderFlowSolver, render_method: str):
        self.flowsolver = flowsolver
        self._init_renderer(render_method=render_method)

    def _init_renderer(self, render_method):
        self.first_time_render = True
        self.render_method = render_method
        self.t_render = []

        # Color limits
        self.VMIN = 0.0
        self.VMAX = 1.5
        # Colormap
        self.cmap = "viridis"
        # Figure aspect ratio
        self.figure_H = 20
        self.figure_W = 30
        self.figure_ar = (
            self.figure_W / self.figure_H
        )  # original aspect ratio of domain

    def set_plot_options(self, plot: plt.plot):
        plot.set_clim(vmin=self.VMIN, vmax=self.VMAX)
        if self.first_time_render:
            self.colorbar = self.figure.colorbar(plot, ax=self.axes)

    def plot_dolfin(self, u):
        """Use dolfin plot function"""
        self.dolfinplt = dolfin.plot(u, cmap=self.cmap)
        self.set_plot_options(self.dolfinplt)

    def plot_sampled(self, coords, values):
        """Scatter plot of sampled field"""
        if self.first_time_render:
            self.scatter = self.axes.scatter(
                coords[:, 0],
                coords[:, 1],
                c=values,
            )
            self.set_plot_options(self.scatter)
        else:
            self.scatter.set_array(values)

    def plot_as_img(self, coords, values, nx, ny):
        """Image plot of sampled field"""
        Z = values.reshape((nx, ny))
        if self.first_time_render:
            x0, x1 = min(coords[:, 0]), max(coords[:, 0])
            y0, y1 = min(coords[:, 1]), max(coords[:, 1])
            self.imshow = self.axes.imshow(Z, extent=[x0, x1, y0, y1], origin="lower")
            self.set_plot_options(self.imshow)
        else:
            self.imshow.set_array(Z)

    def render(self):
        """Render field with provided options"""
        t00 = time.time()

        # Full field & magnitude
        u = self.flowsolver.fields.U0 + self.flowsolver.fields.u_
        u_mag = dolfin.sqrt(dolfin.dot(u, u))

        # Subdomain of interest
        xbnd, ybnd = (-1, 10), (-2, 2)
        nx = 50
        ny = 50

        if self.first_time_render:
            self.figure, self.axes = plt.subplots()
            self.axes.set_title("Velocity magnitude")
            self.axes.set_xlabel("x")
            self.axes.set_ylabel("y")
            self.axes.set_title("Velocity magnitude")

            fig_w = xbnd[1] - xbnd[0]
            fig_h = ybnd[1] - ybnd[0]
            fig_alpha = 20
            self.figure.set_size_inches(
                w=fig_w * fig_alpha / self.figure_W, h=fig_h * fig_alpha / self.figure_H
            )
            self.figure.tight_layout()
            plt.show()

            mesh = self.flowsolver.mesh
            # mesh_coarse = UFieldProcessor.coarsen_mesh(mesh, nx=nx, ny=ny)
            self.submesh = UFieldProcessor.make_submesh(mesh, xbnd=xbnd, ybnd=ybnd)
            self.target_function_space = dolfin.FunctionSpace(self.submesh, "DG", 0)
            self.projection_operator = ProjectionOperator(
                target_space=self.target_function_space
            )

        # Project and plot
        if self.render_method == "project":
            u_proj = UFieldProcessor.project_to(u_mag, self.target_function_space)
            u_proj = self.projection_operator.project(u_mag)
            self.plot_dolfin(u_proj)

        # # Option B: Sample and plot
        if self.render_method == "sample":
            coords_sampl, u_mag_sampl = UFieldProcessor.sample_on_grid(
                u=u_mag, xbnd=xbnd, ybnd=ybnd, nx=nx, ny=ny
            )
            # sample on mesh vertices: very expensive
            # coords_sampl, u_mag_sampl = UFieldProcessor.evaluate_on_vertices(
            #     u=u_mag, mesh=self.submesh
            # )

            # self.plot_as_img(coords=coords_sampl, values=u_mag_sampl, nx=nx, ny=ny)
            self.plot_sampled(coords=coords_sampl, values=u_mag_sampl)

        self.first_time_render = False

        # Draw
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        # Measure time
        t_render = time.time() - t00
        print(f"Rendered in: {t_render}")
        self.t_render += []

    def close(self):
        if self.figure is not None:
            plt.ioff()
            plt.show()
            print("Closing Renderer")
            plt.close()


#########################################################################
#########################################################################
#########################################################################
class UFieldProcessor:
    @staticmethod
    def project_to(u: dolfin.Function, function_space: dolfin.FunctionSpace):
        """
        Project velocity field to function_space (CG1, DG0...)
        """
        V_proj = flu.projectm(u, function_space)
        return V_proj

    @staticmethod
    def sample_on_grid(
        u: dolfin.Function,
        xbnd: tuple,
        ybnd: tuple,
        nx: int,
        ny: int,
    ):
        """
        Sample velocity field on a regular grid.

        Returns:
            coords: array of shape (n_valid_points, 2) - coordinates
            values: array of shape (n_valid_points, 1) - velocity mag
        """
        # Get mesh bounds
        x_min, x_max = xbnd
        y_min, y_max = ybnd

        # Create regular grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)

        # Sample velocity at grid points
        points = np.column_stack([X.ravel(), Y.ravel()])
        values = []
        coords = []

        for point in points:
            try:
                vel = u([point])
            except Exception:
                vel = np.nan
            values.append(vel)
            coords.append(point)

        return np.array(coords), np.array(values)

    @staticmethod
    def evaluate_on_vertices(u, mesh):
        """
        Sample velocity field at mesh vertices.

        Returns:
            coords: array of shape (n_vertices, 2) - vertex coordinates
            values: array of shape (n_vertices, 1) - velocity magnitude
        """
        coords = mesh.coordinates()
        values = np.zeros((len(coords),))

        for i, coord in enumerate(coords):
            try:
                values[i] = u([coord])
            except Exception:
                values[i] = np.nan

        return coords, values

    @staticmethod
    def get_mesh_bnds(mesh):
        x_min, x_max = (
            mesh.coordinates()[:, 0].min(),
            mesh.coordinates()[:, 0].max(),
        )

        y_min, y_max = (
            mesh.coordinates()[:, 1].min(),
            mesh.coordinates()[:, 1].max(),
        )
        return (x_min, x_max), (y_min, y_max)

    @staticmethod
    def coarsen_mesh(mesh, nx=10, ny=10):
        (x_min, x_max), (y_min, y_max) = UFieldProcessor.get_mesh_bnds(mesh)

        coarse_mesh = dolfin.RectangleMesh(
            dolfin.Point(x_min, y_min), dolfin.Point(x_max, y_max), nx, ny
        )
        return coarse_mesh

    @staticmethod
    def make_coarse_mesh(mesh, resolution=64):
        import mshr  # should fail

        xbnd, ybnd = UFieldProcessor.get_mesh_bnds(mesh=mesh)
        # Outer domain: rectangle
        outer = mshr.Rectangle(
            point1=mshr.Point(xbnd(0), ybnd(0)), point2=mshr.Point(xbnd(1), ybnd(1))
        )

        # Hole: circle at center (0.5, 0.5), radius 0.2
        hole = mshr.Circle(center=mshr.Point(0.0, 0.0), r=1)

        # Perform Boolean subtraction
        domain = outer - hole

        # Generate mesh (resolution controls number of cells)
        mesh = mshr.generate_mesh(domain, resolution=resolution)
        return mesh

    @staticmethod
    def make_submesh(mesh, xbnd, ybnd):
        class ViewSubDomain(dolfin.SubDomain):
            def inside(self, x, on_boundary):
                return (
                    x[0] > xbnd[0]
                    and x[0] < xbnd[1]
                    and x[1] > ybnd[0]
                    and x[1] < ybnd[1]
                )

        # Mark subdomain cells
        cell_markers = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim())
        cell_markers.set_all(0)
        ViewSubDomain().mark(cell_markers, 1)

        # Extract submesh
        submesh = dolfin.SubMesh(mesh, cell_markers, 1)

        return submesh


class ProjectionOperator:
    """
    Reusable projection operator
    """

    def __init__(self, target_space: dolfin.FunctionSpace):
        self.target_space = target_space
        self.dx = dolfin.Measure("dx", domain=target_space.mesh())

        # Pre-assemble mass matrix (this is the expensive part)
        u = dolfin.TrialFunction(target_space)
        v = dolfin.TestFunction(target_space)

        self.mass_form = u * v * self.dx
        self.mass_matrix = dolfin.assemble(self.mass_form)
        self.solver = dolfin.LUSolver(self.mass_matrix, "mumps")
        self.result = dolfin.Function(target_space)
        self.rhs_vector = dolfin.Vector()

    def project(self, expression):
        """
        Project expression onto target space using pre-assembled operator
        """
        # Assemble RHS
        v = dolfin.TestFunction(self.target_space)
        rhs_form = expression * v * self.dx
        dolfin.assemble(rhs_form, tensor=self.rhs_vector)

        # Solve using pre-factorized matrix
        self.solver.solve(self.result.vector(), self.rhs_vector)

        return self.result


#########################################################################
#########################################################################
#########################################################################
