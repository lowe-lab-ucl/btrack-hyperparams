import btrack
import json
import optuna
import numpy as np


from collections import namedtuple


def _load_csv():
    objects = btrack.dataio.import_CSV("/Users/arl/Dropbox/Code/py3/BayesianTracker/tests/_test_data/test_data.csv")
    return objects


def _load_ground_truth():
    with open("/Users/arl/Dropbox/Code/py3/BayesianTracker/tests/_test_data/test_ground_truth.json", "r") as file:
        ground_truth = json.load(file)
    return ground_truth


OBJECTS = _load_csv()
GROUND_TRUTH = _load_ground_truth()


def test_tracker(params:dict) -> float:
    """Test the operation of the tracker, using the default config and known
    data."""

    # run the tracking
    with btrack.BayesianTracker() as tracker:
        tracker.configure(params)
        tracker.append(OBJECTS)
        tracker.volume = ((0, 1600), (0, 1200), (-1e5, 1e5))
        tracker.track_interactive(step_size=100)
        tracker.optimize(options={'tm_lim': int(6e4)})
        tracks = tracker.tracks

    # iterate over the tracks and check that the object references match
    accuracy = 0

    for track in tracks:
        try:
            gt_refs = GROUND_TRUTH[str(track.ID)]
            accuracy += sum([ref == gt for ref, gt in zip(track.refs, gt_refs)])
        except KeyError:
            pass

    return accuracy


def suggest_motion(trial, trial_config):

    accuracy = trial.suggest_float('accuracy', 1e-5, 7.5)
    trial_config["MotionModel"]["accuracy"] = accuracy

    prob_not_assign = trial.suggest_float('prob_not_assign', 1e-5, 1e-1, log=True)
    trial_config["MotionModel"]["prob_not_assign"] = prob_not_assign

    max_lost = trial.suggest_int('max_lost', 0, 10)
    trial_config["MotionModel"]["max_lost"] = int(max_lost)


    for key in ["P", "G", "R"]:
        trial_config["MotionModel"][key]["sigma"] = trial.suggest_float('P_sigma', 1e-1, 1e3, log=True)

    return trial_config


def suggest_hypothesis(trial, trial_config):

    lambda_theta = ["lambda_time", "lambda_dist", "lambda_link", "lambda_branch", "theta_dist", "theta_time"]

    for key in lambda_theta:
        trial_config["HypothesisModel"][key] = trial.suggest_float(key, 0.1, 100.)

    rate = ["segmentation_miss_rate", "apoptosis_rate"]

    for key in rate:
        trial_config["HypothesisModel"][key] = trial.suggest_float(key, 0.1, 1., log=True)

    thresh = ["dist_thresh", "time_thresh", "apop_thresh"]

    for key in thresh:
        trial_config["HypothesisModel"][key] = int(trial.suggest_int(key, 0, 10))

    trial_config["HypothesisModel"]["relax"] = bool(trial.suggest_int('relax', 0, 1))

    return trial_config



def objective(trial):

    trial_config = default_config["TrackerConfig"].copy()

    trial_config = suggest_motion(trial, trial_config)
    trial_config = suggest_hypothesis(trial, trial_config)

    params = config(trial_config)
    accuracy = test_tracker(params)

    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))
