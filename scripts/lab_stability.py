"""Lab: why does the training loss diverge? Trains one configuration for a fixed number of epochs (no early
stopping, no test scoring) under one variant and prints the per-epoch curve. Writes nothing to runs/.

  python scripts/lab_stability.py --preset audio_only --split 0 --epochs 12 [--cpu] [--lr 1e-4] [--clipnorm 1]
"""
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--preset", default="audio_only")
parser.add_argument("--split", type=int, default=0)
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--clipnorm", type=float)
parser.add_argument("--cpu", action="store_true", help="hide the Metal GPU")
parser.add_argument("--no-bn", action="store_true", help="replace BatchNormalization by identity")
parser.add_argument("--label", default="")
args = parser.parse_args()

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402

if args.cpu:
    tf.config.set_visible_devices([], "GPU")

from mffer import models  # noqa: E402
from mffer.data.pipeline import Cremad, fit_stats, label_coverage, make_dataset  # noqa: E402
from mffer.data.splits import rows  # noqa: E402

_adam = keras.optimizers.Adam
keras.optimizers.Adam = lambda _lr: _adam(learning_rate=args.lr, clipnorm=args.clipnorm)  # build_model's Adam
if args.no_bn:
    keras.layers.BatchNormalization = lambda **_: keras.layers.Identity()

keras.utils.set_random_seed(args.split)
spec = models.PRESETS[args.preset]
data = Cremad()
train, val = (rows(data.clips, args.split, p) for p in ("train", "val"))
stats = fit_stats(data, train)
model = models.build_model(spec, coverage=label_coverage(data, train, spec))


class Report(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.t = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        weights = max(float(abs(w).max()) for w in self.model.get_layer("displayed_emotion").get_weights())
        print(f"{args.label}\tepoch {epoch + 1:2d}  loss {logs['loss']:7.2f}  "
              f"disp_loss {logs['displayed_emotion_loss']:6.2f}  val_disp_acc {logs['val_displayed_emotion_accuracy']:.3f}  "
              f"val_loss {logs['val_loss']:6.2f}  |w_head|max {weights:6.2f}  {time.perf_counter() - self.t:4.0f}s",
              flush=True)


model.fit(make_dataset(data, train, spec, stats, training=True, seed=args.split),
          validation_data=make_dataset(data, val, spec, stats, training=False),
          epochs=args.epochs, verbose=0, shuffle=False, callbacks=[Report()])
