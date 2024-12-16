import csv
import os
from collections import deque
from typing import Any

from leap_c.util import add_prefix_extend, collect_status


class NumberLogger:
    """Logger to track the data of every step as sums or moving averages and flush it occasionally.
    Only works for simple numerical data at the moment.

    How to operate:
        The only method that has to be used is "log".
        It logs the given stats to internal data. Not committing saves the data intermediately,
        but it will only be written to the main data with the next log with commit_a_step=True
        and count as the same step.
        Logging keys again that have been in a recent log call without committing in between,
        will raise an error to avoid duplicate keys in a step. Every self.flush_frequency-many
        commits will flush the data to a csv file in the given save_directory.
    """

    def __init__(
        self,
        save_directory_path: str,
        save_frequency: int,
        moving_average_width: int,
        keys_to_sum_instead_of_averaging: tuple[str, ...] = (
            "actor_status_0",
            "actor_status_1",
            "actor_status_2",
            "actor_status_3",
            "actor_status_4",
        ),
    ):
        """
        Attributes:
            save_directory_path: The directory where the log file will be saved.
            save_frequency: The frequency at which the data will be saved to the csv file.
            moving_average_width: The width of the moving average.
            keys_to_sum_instead_of_averaging: The keys that should be summed instead of averaged.
        """
        self.filepath = os.path.join(save_directory_path, "log.csv")
        self.uncommitted_data = dict()
        self.data_for_log = dict()
        self.data_behind_the_scenes = dict()  # Contains data for the running averages
        self.save_frequency = save_frequency
        self.commits = 0
        self.moving_average_width = moving_average_width
        self.keys_to_sum_instead_of_averaging = keys_to_sum_instead_of_averaging
        self.latest_fieldnames = None

    def _preprocess_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        status = stats.pop("actor_status", None)
        if status is None:
            return stats
        else:
            status_occurrences = collect_status(status)
            modified_stats = dict()
            for i, num in enumerate(status_occurrences):
                stats[f"actor_status_{i}"] = num
            return {**stats, **modified_stats}

    def _commit_a_step(self, stats_to_commit: dict[str, Any]):
        for key in stats_to_commit:
            if key in self.keys_to_sum_instead_of_averaging:
                self.data_for_log[key] = self.data_for_log.get(
                    key, 0
                ) + stats_to_commit.get(key, 0)
            else:  # Average it
                data_to_commit = stats_to_commit.get(key, None)
                if data_to_commit is not None:
                    key_data = self.data_behind_the_scenes.get(
                        key, (deque(maxlen=self.moving_average_width), 0)
                    )

                    deq = key_data[0]
                    num = key_data[1] + 1
                    deq.append(data_to_commit)
                    self.data_behind_the_scenes["key"] = (deq, num)

                    self.data_for_log[key] = sum(deq) / min(
                        num, self.moving_average_width
                    )

    def log(self, stats: dict[str, Any], commit_a_step: bool) -> dict[str, Any] | None:
        """Returns the current data (moving averages etc.) if the data is being saved to the csv,
        otherwise None.
        See class description on more information how to use."""
        processed_stats = self._preprocess_stats(stats)

        # Throws an error if a key is already in the uncommitted data.
        add_prefix_extend("", self.uncommitted_data, processed_stats)
        if not commit_a_step:
            return None
        else:
            self._commit_a_step(self.uncommitted_data)
            self.commits += 1
            self.uncommitted_data = dict()
            if self.commits % self.save_frequency == 0:
                self._save(self.data_for_log)
                return self.data_for_log  # Return the current data so custom logging can be easily done on top of this.

    def _save(self, data: dict[str, Any]):
        with open(self.filepath, "a") as csv_file:
            data_keys = sorted(data.keys())
            writer = csv.DictWriter(csv_file, fieldnames=data_keys)
            if self.latest_fieldnames != data_keys:
                # Write header when the fieldnames change
                self.latest_fieldnames = data_keys
                writer.writeheader()
            writer.writerow(data)
        self.uncommitted_data = dict()


class WandbLogger(NumberLogger):
    """Very simple logger on top of the NumberLogger that logs to wandb,
    when the NumberLogger saves data to the csv file."""

    def __init__(
        self,
        save_directory_path: str,
        save_frequency: int,
        moving_average_width: int,
        keys_to_sum_instead_of_averaging: tuple[str, ...] = (
            "actor_status_0",
            "actor_status_1",
            "actor_status_2",
            "actor_status_3",
            "actor_status_4",
        ),
    ):
        super().__init__(
            save_directory_path=save_directory_path,
            save_frequency=save_frequency,
            moving_average_width=moving_average_width,
            keys_to_sum_instead_of_averaging=keys_to_sum_instead_of_averaging,
        )
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "To use the WandbLogger, you need to install wandb. You can do this by running pip install wandb."
            )
        if wandb:
            wandb.login()

    def init(
        self,
        project_name: str,
        run_name: str,
        mode: str,
        **kwargs,
    ):
        """Init corresponding to wandb.init(project_name, run_name, mode, **kwargs)."""
        import wandb

        wandb.init(
            project=project_name,
            name=run_name,
            mode=mode,  # type:ignore
            **kwargs,
        )
        self._init_called = True

    def log(self, stats: dict[str, Any], commit: bool):
        # Let NumberLogger do its thing and only log to wandb if step_dict is not None,
        # meaning data is being flushed.
        step_dict = super().log(stats, commit)
        if step_dict is not None:
            import wandb

            if not self._init_called:
                raise ValueError("You need to call init before logging.")
            wandb.log(step_dict)
