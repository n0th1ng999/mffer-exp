"""Time training steps on real data through the input pipeline, and the reading alone.

Answers: how long does one epoch take on this machine, and is reading data the bottleneck?
"""
import argparse
import time
from dataclasses import replace

import yaml

from mffer.data.pipeline import Cremad, fit_stats, label_coverage, make_dataset
from mffer.data.splits import rows
from mffer.models import PRESETS, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="m3")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--set", nargs="*", default=[], metavar="FLAG=VALUE",
                        help="ModelSpec overrides, e.g. --set audio_features=logmel augment=true width=2")
    args = parser.parse_args()

    data = Cremad()
    train = rows(data.clips, 0, "train")
    spec = replace(PRESETS[args.preset], **{k: yaml.safe_load(v) for k, v in (f.split("=", 1) for f in args.set)})
    stats = fit_stats(data, train, spec.audio_features)
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

    print(f"{args.preset} {args.set}: reading {read:.3f} s/batch, training {step:.3f} s/step (reading overlapped)")
    print(f"-> one epoch of {steps_per_epoch} steps ≈ {step * steps_per_epoch / 60:.1f} min; "
          f"90 epochs at most ≈ {90 * step * steps_per_epoch / 3600:.1f} h")


if __name__ == "__main__":
    main()
