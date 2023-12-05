import numpy as np
from obspy.core import Trace

from twistpy.polarization import EstimatorConfiguration
from twistpy.polarization.time import TimeDomainAnalysis6C
from twistpy.polarization.timefrequency import TimeFrequencyAnalysis6C

rng = np.random.default_rng(1)

data = rng.random((20, 6))

traN = Trace(data=data[:, 0], header={"starttime": 0, "delta": 1})
traE = Trace(data=data[:, 1], header={"starttime": 0, "delta": 1})
traZ = Trace(data=data[:, 2], header={"starttime": 0, "delta": 1})
rotN = Trace(data=data[:, 3], header={"starttime": 0, "delta": 1})
rotE = Trace(data=data[:, 4], header={"starttime": 0, "delta": 1})
rotZ = Trace(data=data[:, 5], header={"starttime": 0, "delta": 1})
est_DOT = EstimatorConfiguration(
    wave_types=["R", "L", "P", "SV", "SH"],
    method="DOT",
    use_ml_classification=False,
    vp=(1000, 2000, 200),
    vp_to_vs=(1.7, 2.1, 5),
    vl=(200, 400, 100),
    vr=(200, 400, 100),
    phi=(0, 360, 180),
    xi=(-90, 90, 90),
    theta=(0, 90, 45),
)
est_MUSIC = EstimatorConfiguration(
    wave_types=["R", "L", "P", "SV", "SH"],
    method="MUSIC",
    use_ml_classification=False,
    vp=(1000, 2000, 200),
    vp_to_vs=(1.7, 2.1, 5),
    vl=(200, 400, 100),
    vr=(200, 400, 100),
    phi=(0, 360, 180),
    xi=(-90, 90, 90),
    theta=(0, 90, 45),
)
est_MVDR = EstimatorConfiguration(
    wave_types=["R", "L", "P", "SV", "SH"],
    method="MVDR",
    use_ml_classification=False,
    vp=(1000, 2000, 200),
    vp_to_vs=(1.7, 2.1, 5),
    vl=(200, 400, 100),
    vr=(200, 400, 100),
    phi=(0, 360, 180),
    xi=(-90, 90, 90),
    theta=(0, 90, 45),
)
est_BARTLETT = EstimatorConfiguration(
    wave_types=["R", "L", "P", "SV", "SH"],
    method="BARTLETT",
    use_ml_classification=False,
    vp=(1000, 2000, 200),
    vp_to_vs=(1.7, 2.1, 5),
    vl=(200, 400, 100),
    vr=(200, 400, 100),
    phi=(0, 360, 180),
    xi=(-90, 90, 90),
    theta=(0, 90, 45),
)


def test_tda():
    window = {"window_length_seconds": 5.0, "overlap": 1.0}
    tda = TimeDomainAnalysis6C(
        traN=traN,
        traE=traE,
        traZ=traZ,
        rotN=rotN,
        rotE=rotE,
        rotZ=rotZ,
        scaling_velocity=1.0,
        free_surface=True,
        window=window,
        verbose=False,
    )

    tda.polarization_analysis(estimator_configuration=est_DOT)
    tda.polarization_analysis(estimator_configuration=est_MUSIC)
    tda.polarization_analysis(estimator_configuration=est_MVDR)
    tda.polarization_analysis(estimator_configuration=est_BARTLETT)


def test_tfa():
    window = {"number_of_periods": 1.0, "frequency_extent": 0.01}
    tfa = TimeFrequencyAnalysis6C(
        traN=traN,
        traE=traE,
        traZ=traZ,
        rotN=rotN,
        rotE=rotE,
        rotZ=rotZ,
        scaling_velocity=1.0,
        free_surface=True,
        window=window,
        verbose=False,
    )

    tfa.polarization_analysis(estimator_configuration=est_DOT)
    tfa.polarization_analysis(estimator_configuration=est_MUSIC)
    tfa.polarization_analysis(estimator_configuration=est_MVDR)
    tfa.polarization_analysis(estimator_configuration=est_BARTLETT)
