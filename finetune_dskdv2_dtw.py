import os
import sys

REPO_DIR = os.path.dirname(__file__)
DSKDV2_CODE_DIR = os.path.join(REPO_DIR, "DSKDv2", "code")
if DSKDV2_CODE_DIR not in sys.path:
    sys.path.insert(0, DSKDV2_CODE_DIR)

from arguments import get_args
import distillation


CRITERION_NAME = "dual_space_kd_v2_with_dtw"


def _force_criterion_in_argv():
    cleaned_args = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--criterion":
            skip_next = True
            continue
        if arg.startswith("--criterion="):
            continue
        cleaned_args.append(arg)
    cleaned_args.extend(["--criterion", CRITERION_NAME])
    sys.argv = [sys.argv[0]] + cleaned_args


def _validate_required_args(args):
    if args.teacher_model_path is None:
        raise ValueError("DSKDv2+DTW requires --teacher-model-path.")
    if args.dtw_weight <= 0:
        raise ValueError("DSKDv2+DTW requires --dtw-weight > 0.")


def main():
    _force_criterion_in_argv()
    args = get_args()
    args.criterion = CRITERION_NAME
    _validate_required_args(args)
    distillation.main()


if __name__ == "__main__":
    main()
