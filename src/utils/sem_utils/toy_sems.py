from collections import OrderedDict
import numpy as np
from src.utils.utilities import sigmoid,expit

class StationaryDependentSEM:
    @staticmethod
    def static():

        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: 4 + np.exp(-sample["X"][t]) + noise
        Y = lambda noise, t, sample: np.cos(sample["Z"][t]) - np.exp(-sample["Z"][t] / 20.0) + noise
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():

        # We get temporal innovation by introducing transfer functions between temporal indices
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: np.exp(-sample["X"][t]) + sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: np.cos(sample["Z"][t])
            - np.exp(-sample["Z"][t] / 20.0)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

class HealthcareSEM:

    @staticmethod
    def static():
        A = lambda noise, time, sample : noise # note this is UNIFORM noise
        B = lambda noise, time, sample: (27. - 0.01 * sample["A"]) + noise
        R = lambda noise, time, sample: sigmoid(-8. + 0.1 * sample["A"] + 0.03 * sample["B"])
        S = lambda noise, time, sample: sigmoid(-13.0 + 0.1 * sample["A"] + 0.2 * sample["B"])
        C = lambda noise, time, sample: sigmoid(2.2 + - 0.05 * sample["A"] + 0.01 * sample["B"] - 0.04 * sample["S"] +  0.02 * sample["R"])
        Y = lambda noise, time, sample: 6.8 + 0.04 * sample["A"] + 0.15 * sample["B"] - 0.6 * sample["S"] + 0.55 * sample["R"] + sample["C"] + noise
        return OrderedDict([("A", A), ("B", B), ("R", R), ("S", S), ("C", C), ("Y", Y)])

    @staticmethod
    def dynamic():
        return None

class EpidemiologySEM:

    @staticmethod
    def static():
        U = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
        T = lambda noise, time, sample: noise  # note this is UNIFORM noise [4,8]
        L = lambda noise, time, sample: expit(0.5 * sample["T"] + sample["U"])
        R = lambda noise, time, sample: 4 + sample["L"]  * sample["T"]
        Y = lambda noise, time, sample: 0.5 + np.cos( 4 * sample["T"]) + np.sin(- sample["L"] + 2 * sample["R"]) + sample["U"] + noise
        return OrderedDict([("U", U), ("T", T), ("L", L), ("R", R), ("Y", Y)])
    # def static():
    #     U = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
    #     T = lambda noise, time, sample: noise  # note this is UNIFORM noise [4,8]
    #     L = lambda noise, time, sample: 0.5 * sample["T"] + sample["U"]
    #     R = lambda noise, time, sample: expit(4 + sample["L"]  * sample["T"])
    #     Y = lambda noise, time, sample: 0.5 + np.cos( 4 * sample["T"] + np.sin(- sample["L"] + 2 * sample["R"])) + sample["U"] + noise
    #     return OrderedDict([("U", U), ("T", T), ("L", L), ("R", R), ("Y", Y)])

    @staticmethod
    def dynamic():
        return None

class ExtendedEpidemiologySEM:

    @staticmethod
    def static():
        P = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
        M = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
        E = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
        Z = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]
        X = lambda noise, time, sample: noise  # note this is UNIFORM noise [-1,1]

        U = lambda noise, time, sample: noise + sample["P"] # note this is UNIFORM noise [-1,1]
        T = lambda noise, time, sample: noise + sample["M"]  # note this is UNIFORM noise [4,8]
        L = lambda noise, time, sample: expit(0.5 * sample["T"] + sample["U"]) + sample["E"]
        R = lambda noise, time, sample: 4 + sample["L"]  * sample["T"] + sample["Z"]
        Y = lambda noise, time, sample: 0.5 + np.cos( 4 * sample["T"]) + np.sin(- sample["L"] + 2 * sample["R"]) + sample["U"] +  sample["X"] + noise
        return OrderedDict([("P", P), ("M", M), ("E", E), ("Z", Z), ("X", X), ("U", U), ("T", T), ("L", L), ("R", R), ("Y", Y)])

    @staticmethod
    def dynamic():
        return None



class StationaryIndependentSEM:
    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    @staticmethod
    def dynamic():
        X = lambda noise, t, sample: -sample["X"][t - 1] + noise
        Z = lambda noise, t, sample: -sample["Z"][t - 1] + noise
        Y = (
            lambda noise, t, sample: -2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
            - np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            + sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryDependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.

    with: intervention_domain = {'X':[-4,1],'Z':[-3,3]}
    """

    def __init__(self, change_point):
        """
        Initialise change point(s).

        Parameters
        ----------
        cp : int
            The temporal index of the change point (cp).
        """
        self.cp = change_point

    @staticmethod
    def static():
        """
        noise: e
        sample: s
        time index: t
        """
        X = lambda e, t, s: e
        Z = lambda e, t, s: s["X"][t] + e
        Y = lambda e, t, s: np.sqrt(abs(36 - (s["Z"][t] - 1) ** 2)) + 1 + e
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    def dynamic(self):
        X = lambda e, t, s: s["X"][t - 1] + e
        Z = (
            lambda e, t, s: -s["X"][t] / s["X"][t - 1] + s["Z"][t - 1] + e
            if t == self.cp
            else s["X"][t] + s["Z"][t - 1] + e
        )
        Y = (
            lambda e, t, s: s["Z"][t] * np.cos(np.pi * s["Z"][t]) - s["Y"][t - 1] + e
            if t == self.cp
            else abs(s["Z"][t]) - s["Y"][t - 1] - s["Z"][t - 1] + e
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])


class NonStationaryIndependentSEM:
    """
    This SEM currently supports one change point.

    This SEM changes topology over t.
    """

    def __init__(self, change_point):
        self.change_point = change_point

    @staticmethod
    def static():
        X = lambda noise, t, sample: noise
        Z = lambda noise, t, sample: noise
        Y = (
            lambda noise, t, sample: -(
                2 * np.exp(-((sample["X"][t] - 1) ** 2) - (sample["Z"][t] - 1) ** 2)
                + np.exp(-((sample["X"][t] + 1) ** 2) - sample["Z"][t] ** 2)
            )
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])

    def dynamic(self):
        #  X_t | X_{t-1}
        X = lambda noise, t, sample: sample["X"][t - 1] + noise
        Z = (
            lambda noise, t, sample: np.cos(sample["Z"][t - 1]) + noise
            if t == self.change_point
            else np.sin(sample["Z"][t - 1] ** 2) * sample["X"][t - 1] + noise
        )
        #  if t <= 1: Y_t | Z_t, Y_{t-1} else: Y_t | Z_t, X_t, Y_{t-1}
        Y = (
            lambda noise, t, sample:
            # np.exp(-np.cos(sample["X"][t]) ** 2)
            -np.exp(-(sample["Z"][t]) / 3.0)
            + np.exp(-sample["X"][t] / 3.0)
            + sample["Y"][t - 1]
            + sample["X"][t - 1]
            + noise
            if t == self.change_point
            else -2 * np.exp(-((sample["X"][t]) ** 2) - (sample["Z"][t] - sample["Z"][t - 1]) ** 2)
            - np.exp(-((sample["X"][t] - sample["Z"][t]) ** 2))
            + np.cos(sample["Z"][t])
            - sample["Y"][t - 1]
            + noise
        )
        return OrderedDict([("X", X), ("Z", Z), ("Y", Y)])
