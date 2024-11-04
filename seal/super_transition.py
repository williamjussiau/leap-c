from dataclasses import dataclass

import numpy as np
import torch
from acados_template.acados_ocp_iterate import AcadosOcpIterate


@dataclass
class SuperTransition:
    """Data class for managing batch of the system which might include much more than the fields of the transition (x, u, r, x_next, done)."""

    x: np.ndarray
    u: np.ndarray
    r: float
    x_next: np.ndarray
    done: bool
    p_global: np.ndarray | None
    p_stagewise: np.ndarray | None
    acados_iterate: AcadosOcpIterate | None

class LazySuperTransitionBatch:
    """Lazy batch for SuperTransitions.
    Since depending on the scenario only some fields of the SuperTransition are actually relevant,
    this tries to save compute by only batching properties when they are accessed.
    NOTE: Note that most properties are Tensors of type float32,
    even though higher precision might be used in the AcadosOcpIterates within the SuperTransitions.
    """

    def __init__(
        self,
        supertransitions: list[SuperTransition],
        device: str,
    ):
        self.transitions = supertransitions
        self.device = device

        self.traditional_transition_part_batched = False
        # Traditional RL fields
        self._x = None
        self._u = None
        self._r = None
        self._x_next = None
        self._done = None

        # Ocp parameter fields
        self._p_global = None
        self._p_stagewise = None

        # Acados Iterate fields
        self._x_traj = None
        self._u_traj = None
        self._z_traj = None  # algebraic vars
        self._sl_traj = None  # lower slack vars
        self._su_traj = None  # upper slack vars
        self._pi_traj = None  # dual vars for dynamics
        self._lam_traj = None  # dual vars for inequalities

        # In some cases you probably just want the iterates themselves
        self._iterates = None

    @property
    def x(self) -> torch.Tensor:
        if self._x is None:
            self._x, self._u, self._r, self._x_next, self._done = (
                self._batch_traditional_transition_part()
            )
        return self._x

    @property
    def u(self) -> torch.Tensor:
        if self._u is None:
            self._x, self._u, self._r, self._x_next, self._done = (
                self._batch_traditional_transition_part()
            )
        return self._u

    @property
    def r(self) -> torch.Tensor:
        if self._r is None:
            self._x, self._u, self._r, self._x_next, self._done = (
                self._batch_traditional_transition_part()
            )
        return self._r

    @property
    def x_next(self) -> torch.Tensor:
        if self._x_next is None:
            self._x, self._u, self._r, self._x_next, self._done = (
                self._batch_traditional_transition_part()
            )
        return self._x_next

    @property
    def done(self) -> torch.Tensor:
        if self._done is None:
            self._x, self._u, self._r, self._x_next, self._done = (
                self._batch_traditional_transition_part()
            )
        return self._done

    @property
    def p_global(self) -> torch.Tensor:
        if self._p_global is None:
            self._batch_p_global()

        return self._p_global  # type: ignore

    @property
    def p_stagewise(self) -> torch.Tensor:
        if self._p_stagewise is None:
            self._batch_p_stagewise()

        return self._p_stagewise  # type: ignore

    @property
    def x_traj(self) -> torch.Tensor:
        if self._x_traj is None:
            self._x_traj = self._batch_acados_iterate_field("x_traj")
        return self._x_traj

    @property
    def u_traj(self) -> torch.Tensor:
        if self._u_traj is None:
            self._u_traj = self._batch_acados_iterate_field("u_traj")
        return self._u_traj

    @property
    def z_traj(self) -> torch.Tensor:
        if self._z_traj is None:
            self._z_traj = self._batch_acados_iterate_field("z_traj")
        return self._z_traj

    @property
    def sl_traj(self) -> torch.Tensor:
        if self._sl_traj is None:
            self._sl_traj = self._batch_acados_iterate_field("sl_traj")
        return self._sl_traj

    @property
    def su_traj(self) -> torch.Tensor:
        if self._su_traj is None:
            self._su_traj = self._batch_acados_iterate_field("su_traj")
        return self._su_traj

    @property
    def pi_traj(self) -> torch.Tensor:
        if self._pi_traj is None:
            self._pi_traj = self._batch_acados_iterate_field("pi_traj")
        return self._pi_traj

    @property
    def lam_traj(self) -> torch.Tensor:
        if self._lam_traj is None:
            self._lam_traj = self._batch_acados_iterate_field("lam_traj")
        return self._lam_traj

    @property
    def iterates(self) -> list[AcadosOcpIterate]:
        if self._iterates is None:
            iterates = []
            for t in self.transitions:
                if t.acados_iterate is None:
                    raise AttributeError(
                        "acados_iterate is not set for the transition."
                    )
                iterates.append(t.acados_iterate)
            self._iterates = iterates
        return self._iterates

    def _batch_traditional_transition_part(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for t in self.transitions:
            s_lst.append(t.x)
            a_lst.append(t.u)
            r_lst.append(t.r)
            s_prime_lst.append(t.x_next)
            done_lst.append(t.done)
        x = torch.tensor(np.array(s_lst, dtype=np.float32), device=self.device)
        u = torch.tensor(np.array(a_lst, dtype=np.float32), device=self.device)
        r = torch.tensor(np.array(r_lst, dtype=np.float32), device=self.device)
        x_next = torch.tensor(
            np.array(s_prime_lst, dtype=np.float32), device=self.device
        )
        done = torch.tensor(np.array(done_lst, dtype=np.float32), device=self.device)
        return x, u, r, x_next, done

    def _batch_p_global(self):
        p_global_lst = []
        for t in self.transitions:
            if t.p_global is None:
                raise AttributeError("p_global is not set for the transition.")
            p_global_lst.append(t.p_global)
        self._p_global = np.array(p_global_lst, dtype=np.float32)

    def _batch_p_stagewise(self):
        p_stagewise_lst = []
        for t in self.transitions:
            if t.p_stagewise is None:
                raise AttributeError("p_stagewise is not set for the transition.")
            p_stagewise_lst.append(t.p_stagewise)
        self._p_stagewise = np.array(p_stagewise_lst, dtype=np.float32)

    def _batch_acados_iterate_field(self, field_name: str) -> torch.Tensor:
        acados_iterate_lst = []
        for t in self.transitions:
            if t.acados_iterate is None:
                raise AttributeError("acados_iterate is not set for the transition.")
            acados_iterate_lst.append(
                self.map_field_name_to_single_phase_iterate_field(
                    field_name, t.acados_iterate
                )
            )
        return torch.tensor(
            np.array(acados_iterate_lst, dtype=np.float32), device=self.device
        )

    def map_field_name_to_single_phase_iterate_field(
        self, field_name: str, iterate: AcadosOcpIterate
    ) -> np.ndarray:
        if field_name == "x_traj":
            field = iterate.x_traj
        elif field_name == "u_traj":
            field = iterate.u_traj
        elif field_name == "z_traj":
            field = iterate.z_traj
        elif field_name == "sl_traj":
            field = iterate.sl_traj
        elif field_name == "su_traj":
            field = iterate.su_traj
        elif field_name == "pi_traj":
            field = iterate.pi_traj
        elif field_name == "lam_traj":
            field = iterate.lam_traj
        if len(field) > 1:
            raise ValueError(
                f"Field {field_name} has varying stage-wise dimensions. Multi-phase OCPs are not supported like this."
            )
        return field[0]

    def __len__(self):
        return len(self.transitions)
