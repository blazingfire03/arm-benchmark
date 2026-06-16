import csv
import json
import os

COLUMNS = ["trial_id", "sub_phase", "algorithm", "test_case_id", "trial_number", "random_seed",
           "planning_success", "execution_success", "path_planning_time", "path_execution_time",
           "total_time", "path_smoothness_msj", "path_distance_raw", "path_distance_optimal",
           "path_optimality_ratio", "collision_flag", "num_contact_points",
           "algorithm_specific_metadata"]


class TrialLogger:
    def __init__(self, path):
        self.path = path
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.writer = csv.DictWriter(self.f, fieldnames=COLUMNS, extrasaction="ignore")
        if new:
            self.writer.writeheader()
            self.f.flush()

    def log(self, row):
        r = dict(row)
        meta = r.get("algorithm_specific_metadata")
        if isinstance(meta, (dict, list)):
            r["algorithm_specific_metadata"] = json.dumps(meta, default=float)
        self.writer.writerow(r)
        self.f.flush()

    def close(self):
        self.f.close()
