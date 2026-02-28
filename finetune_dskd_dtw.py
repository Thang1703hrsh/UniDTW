import os
import sys

REPO_DIR = os.path.dirname(__file__)
DSKD_CODE_DIR = os.path.join(REPO_DIR, "DSKD", "code")
if DSKD_CODE_DIR not in sys.path:
    sys.path.insert(0, DSKD_CODE_DIR)

from arguments import get_args
import distillation


CRITERION_NAME = "min_edit_dis_kld"


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
        raise ValueError("DSKD+DTW requires --teacher-model-path.")
    if args.teacher_to_student_id_mapping is None:
        raise ValueError(
            "DSKD+DTW requires --teacher-to-student-id-mapping (JSON).")
    if not os.path.exists(args.teacher_to_student_id_mapping):
        raise ValueError(
            f"Mapping file not found: {args.teacher_to_student_id_mapping}")


def main():
    _force_criterion_in_argv()
    args = get_args()
    args.criterion = CRITERION_NAME
    _validate_required_args(args)
    distillation.main()


if __name__ == "__main__":
    main()
