"""Time training steps on real data through the input pipeline, and the reading alone.

Answers: how long does one epoch take on this machine, and is reading data the bottleneck?
"""
import argparse
import time

from mffer.data.pipeline import Cremad, fit_stats, label_coverage, make_dataset
from mffer.data.splits import rows
from mffer.models import PRESETS, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="m3")
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()

    data = Cremad()
    train = rows(data.clips, 0, "train")
    spec = PRESETS[args.preset]
    stats = fit_stats(data, train)
    steps_per_epoch = -(-len(train) // 16)

    dataset = make_dataset(data, train, spec, stats, training=True)
    start = time.perf_counter()
    for _ in dataset.take(args.steps):
        pass
    read = (time.perf_counter() - start) / args.steps

    model = build_model(spec, coverage=label_coverage(data, train, spec))
    model.fit(dataset.take(5), verbose=0)                       # compile the graph
    start = time.perf_counter()
    model.fit(make_dataset(data, train, spec, stats, training=True, seed=1).take(args.steps), verbose=0)
    step = (time.perf_counter() - start) / args.steps

    print(f"{args.preset}: reading {read:.3f} s/batch, training {step:.3f} s/step (reading overlapped)")
    print(f"-> one epoch of {steps_per_epoch} steps ≈ {step * steps_per_epoch / 60:.1f} min; "
          f"30 epochs at most ≈ {30 * step * steps_per_epoch / 3600:.1f} h")


if __name__ == "__main__":
    main()
