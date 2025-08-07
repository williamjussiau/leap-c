import time

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import utils.utils_flowsolver as flu
from examples.cylinder.cylinderflowsolver import CylinderFlowSolver

plt.ion()


class CylinderRenderer:
    def __init__(
        self, flowsolver: CylinderFlowSolver, render_method: str, render_mode: str
    ):
        self.flowsolver = flowsolver
        self.render_method = render_method
        self.render_mode = render_mode
        self._init_renderer()

    def _init_renderer(self):
        self.first_time_render = True
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

    def _first_render(self):
        # Figure
        self.figure, self.axes = plt.subplots()
        self.axes.set_xlabel("x")
        self.axes.set_ylabel("y")
        self.axes.set_title("Velocity magnitude")

        # Subdomain of interest
        self.xbnd, self.ybnd = (-1, 10), (-3, 3)
        self.nx = 50
        self.ny = 50
        fig_w = self.xbnd[1] - self.xbnd[0]
        fig_h = self.ybnd[1] - self.ybnd[0]
        fig_alpha = 20
        self.figure.set_size_inches(
            w=fig_w * fig_alpha / self.figure_W, h=fig_h * fig_alpha / self.figure_H
        )
        self.figure.tight_layout()
        plt.show()

        # Submesh, projection and samping
        mesh = self.flowsolver.mesh
        # mesh_coarse = UFieldProcessor.coarsen_mesh(mesh, nx=nx, ny=ny)
        self.submesh = UFieldProcessor.make_submesh(
            mesh, xbnd=self.xbnd, ybnd=self.ybnd
        )
        CG1 = dolfin.FunctionSpace(self.submesh, "CG", 1)  # Continuous Galerkin d°1
        DG0 = dolfin.FunctionSpace(self.submesh, "DG", 0)  # Discontinuous Galerkin d°0
        self.target_function_space = CG1
        self.projection_operator = ProjectionOperator(
            target_space=self.target_function_space
        )
        # linspace sampling grid
        self.sampling_grid = UFieldProcessor.make_sampling_grid(
            self.xbnd, self.ybnd, self.nx, self.ny
        )
        # or mesh coordinates (expensive)
        # self.sampling_grid = mesh.coordinates()
        _, self.U0_sampl = UFieldProcessor.sample_at(
            u=self.flowsolver.fields.U0, points=self.sampling_grid
        )

        # Indexing + get_local
        # Manual function assigner
        W_mixfuncspace = self.flowsolver.W
        alldof_coord = W_mixfuncspace.tabulate_dof_coordinates()  # coord of all dof
        alldof_idx = flu.get_subspace_dofs(W_mixfuncspace)  # idx of u, v, p dof
        udof_coord = alldof_coord[alldof_idx["u"], :]  # coord of u dof
        vdof_coord = alldof_coord[alldof_idx["v"], :]  # coord of v dof
        # cut subdomain
        x_min, x_max = self.xbnd
        y_min, y_max = self.ybnd

        def mask_subdomain(array):
            return (
                (array[:, 0] >= x_min)
                & (array[:, 0] <= x_max)
                & (array[:, 1] >= y_min)
                & (array[:, 1] <= y_max)
            )

        mask_udof = mask_subdomain(udof_coord)
        mask_vdof = mask_subdomain(vdof_coord)
        udof_coord = udof_coord[mask_udof]
        vdof_coord = vdof_coord[mask_vdof]
        alldof_idx["u"] = alldof_idx["u"][mask_udof]
        alldof_idx["v"] = alldof_idx["v"][mask_vdof]
        # now sort coordinates to have same ordering for u, v
        udof_sort_idx = np.lexsort((udof_coord[:, 1], udof_coord[:, 0]))
        vdof_sort_idx = np.lexsort((vdof_coord[:, 1], vdof_coord[:, 0]))
        udof_coord_sorted = udof_coord[udof_sort_idx]
        # vdof_coord_sorted = vdof_coord[vdof_sort_idx] # equiv to prev line
        # extract from get_local()
        UP0 = self.flowsolver.fields.UP0.vector().get_local()
        u0 = UP0[alldof_idx["u"]][udof_sort_idx]
        v0 = UP0[alldof_idx["v"]][udof_sort_idx]

        self.u0_values_sorted = u0
        self.v0_values_sorted = v0
        self.udof_sort_idx = udof_sort_idx
        self.vdof_sort_idx = vdof_sort_idx
        self.dof_coord_sorted = udof_coord_sorted
        self.alldof_idx = alldof_idx

    def render(self):
        """Render field with provided options"""
        t00 = time.time()

        if self.first_time_render:
            self._first_render()

        # # Project
        # if self.render_method == "project":
        #     u_vec = self.flowsolver.fields.u_ + self.flowsolver.fields.U0
        #     u_mag = dolfin.sqrt(dolfin.dot(u_vec, u_vec))
        #     # u_proj = UFieldProcessor.project_to(u_mag, self.target_function_space)
        #     u_proj = self.projection_operator.project(u_mag)
        #     self.plot_dolfin(u_proj)

        # # Sample
        # if self.render_method == "sample":
        #     u_vec = self.flowsolver.fields.u_
        #     _, u_sampl = UFieldProcessor.sample_at(u=u_vec, points=self.sampling_grid)
        #     u_mag_sampl = np.linalg.norm(u_sampl + self.U0_sampl, ord=2, axis=1)

        #     # self.plot_sampled(coords=self.sampling_grid, values=u_mag_sampl)
        #     self.plot_as_img(
        #         coords=self.sampling_grid, values=u_mag_sampl, nx=self.nx, ny=self.ny
        #     )

        # # Reindexing
        # if self.render_method == "index":
        #     if self.flowsolver.fields.up_ is None:
        #         self.first_time_render = False
        #         return 1  # fast exit
        #     up_vec = self.flowsolver.fields.up_.vector().get_local()
        #     u_values_sorted = up_vec[self.alldof_idx["u"]][self.udof_sort_idx]
        #     v_values_sorted = up_vec[self.alldof_idx["v"]][self.vdof_sort_idx]
        #     U_values_sorted = np.vstack(
        #         (
        #             u_values_sorted + self.u0_values_sorted,
        #             v_values_sorted + self.v0_values_sorted,
        #         )
        #     ).T
        #     u_mag_sorted = np.linalg.norm(U_values_sorted, ord=2, axis=1)
        #     self.plot_sampled(coords=self.dof_coord_sorted, values=u_mag_sorted)

        # Draw
        # self.figure.canvas.draw()
        # self.figure.canvas.flush_events()
        # if self.render_mode == "rgb_array":
        #     u_rgb = np.asarray(self.figure.canvas.renderer.buffer_rgba())[:, :, :3]
        # else:
        #     u_rgb = 0
        u_rgb = 0

        # Measure time
        t_render = time.time() - t00
        # print(f"Rendered in: {t_render}")
        self.t_render.append(t_render)
        self.first_time_render = False
        return u_rgb

    def close(self):
        if self.figure is not None:
            print("Closing Renderer")
            plt.close("all")
        return 0


#########################################################################
#########################################################################
#########################################################################
class UFieldProcessor:
    @staticmethod
    def project_to(u: dolfin.Function, function_space: dolfin.FunctionSpace):
        """
        Project velocity field to function_space (CG1, DG0...)
        """
        u_proj = flu.projectm(u, function_space)
        return u_proj

    @staticmethod
    def sample_at(u: dolfin.Function, points: np.ndarray, method: str = "eval"):
        npoints = points.shape[0]
        values = np.zeros((npoints, 2))

        nan_array = np.array([np.nan, np.nan])
        if method == "eval":
            for i, point in enumerate(points):
                try:
                    u.eval(values[i, :], point)
                except Exception:
                    values[i, :] = nan_array
        else:
            for i, point in enumerate(points):
                try:
                    values[i, :] = u([point])
                except Exception:
                    values[i, :] = nan_array

        return np.array(points), np.array(values)

    @staticmethod
    def make_sampling_grid(xbnd, ybnd, nx, ny):
        # Get mesh bounds
        x_min, x_max = xbnd
        y_min, y_max = ybnd

        # Create regular grid
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        return points

    @staticmethod
    def evaluate_on_vertices(u, mesh):
        """
        Sample velocity field at mesh vertices.

        Returns:
            coords: array of shape (n_vertices, 2) - vertex coordinates
            values: array of shape (n_vertices, 1) - velocity magnitude
        """
        return UFieldProcessor.sample_at(u=u, points=mesh.coordinates())

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
