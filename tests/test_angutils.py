import unittest
import numpy as np
from angutils.angutils import (
    calcDCM,
    DCM2axang,
    rotMat,
    skew,
    colVec,
    vnorm,
    calcang,
    projplane,
    EulerAng2DCM,
    DCM2EulerAng,
)


class TestAngUtils(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calcDCM(self):
        """Test for expected outputs for the three simple DCMs about body axes"""

        ns = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        th = 0.5
        for j, n in enumerate(ns):
            self.assertTrue(
                np.max(np.abs(calcDCM(n, th).T - rotMat(j + 1, th))) < 1e-15
            )

    def test_DCM_roundtrip(self):
        """Test that DCM2axang and calcDCM are true inverses of one another"""

        # generate random rotation axes and angles
        N = int(1000)
        ns = np.random.randn(3, N)
        norms = np.sqrt(np.sum(ns**2, axis=0))
        for j in range(3):
            ns[j] = ns[j] / norms

        ths = np.random.rand(N) * 2 * np.pi

        # loop through and check round-trip calculation
        thtols = np.zeros(N) + 1e-15
        thtols[ths < 16 * np.pi / 180] = 1e-10
        thtols[ths > (2 - 16 / 180) * np.pi] = 1e-10
        thtols[np.abs(ths - np.pi) < 35 * np.pi / 180] = 1e-14
        thtols[np.abs(ths - np.pi) < 5 * np.pi / 180] = 1e-10

        ntols = np.zeros(N) + 1e-14
        ntols[ths < 10 * np.pi / 180] = 1e-5
        ntols[ths > (2 - 10 / 180) * np.pi] = 1e-5
        ntols[np.abs(ths - np.pi) < 15 * np.pi / 180] = 1e-12
        ntols[np.abs(ths - np.pi) < 2 * np.pi / 180] = 1e-5

        for j, (n, th, ntol, thtol) in enumerate(zip(ns.T, ths, ntols, thtols)):
            DCM = calcDCM(n, th)
            n1, th1 = DCM2axang(DCM)

            # two pass conditions:
            if th <= np.pi:
                # expect everything to be the same
                self.assertTrue(np.abs(th - th1) < thtol)
                self.assertTrue(np.all(np.abs(n - n1) < ntol))
            else:
                # expect angle to be 2\pi - th and axis to be negative
                self.assertTrue(np.abs(th - (2 * np.pi - th1)) < thtol)
                self.assertTrue(np.all(np.abs(n + n1) < ntol))

    def test_EulerAng_roundtrip(self):
        """Test that EulerAng2DCM and DCM2EulerAng are true inverses of one another"""

        # generate random rotation sets
        N = int(1000)

        # pick 1st and 3rd rots randomly and then ensure that second rotation is not
        # redundant with either
        rotSets = np.zeros((N, 3), dtype=int)
        rotSets[:, 0] = np.random.choice([1, 2, 3], N)
        rotSets[:, 2] = np.random.choice([1, 2, 3], N)
        totset = set([1, 2, 3])
        for j in range(N):
            rotSets[j, 1] = np.random.choice(list(totset - set(rotSets[j, [0, 2]])), 1)[
                0
            ]

        # generate random rotation angles
        angSets = np.random.rand(N, 3) * 2 * np.pi - np.pi

        # check the roundtrip
        tol = 1e-15
        for j, (rotSet, angs) in enumerate(zip(rotSets, angSets)):
            # body:
            DCM = EulerAng2DCM(rotSet, angs)
            angsout = DCM2EulerAng(DCM, rotSet)
            DCM2 = EulerAng2DCM(rotSet, angsout)
            self.assertTrue(np.max(np.abs(DCM - DCM2)) < tol)

            # space:
            DCM = EulerAng2DCM(rotSet, angs, body=False)
            angsout = DCM2EulerAng(DCM, rotSet, body=False)
            DCM2 = EulerAng2DCM(rotSet, angsout, body=False)
            self.assertTrue(np.max(np.abs(DCM - DCM2)) < tol)

    def test_skew(self):
        """Test skew-symmetric property for random inputs"""

        tmp = skew(np.random.rand(3, 1))

        should_be_zeros = tmp + tmp.T

        self.assertTrue(np.all(should_be_zeros == 0))

    def test_colVec(self):
        """Test output dimensionality for random input"""

        self.assertTrue(colVec(np.random.rand(1, 3)).shape == (3, 1))
        self.assertTrue(colVec(np.random.rand(3, 1)).shape == (3, 1))
        self.assertTrue(colVec(np.random.rand(3).tolist()).shape == (3, 1))

    def test_vnorm(self):
        """Test norm output for random inputs"""

        self.assertTrue(np.abs(np.linalg.norm(vnorm(np.random.rand(1, 3))) - 1) < 1e-15)

    def test_calcang(self):
        """Test computed angles for random inputs"""

        x = vnorm(colVec([1, 0, 0]))
        z = vnorm(colVec([0, 0, 1]))
        th = np.pi / 6
        y = np.matmul(calcDCM(z, th), x)

        self.assertTrue(np.abs(calcang(x, y, z) - th) < 1e-15)

        for j in range(int(1e4)):
            x = vnorm(colVec(np.random.randn(3)))
            z = np.random.randn(2)
            z = vnorm(colVec(np.hstack([z, -np.matmul(z, x[:2]) / x[2]])))
            self.assertTrue(np.abs(np.matmul(x.T, z)) < 1e-15)
            th = (np.random.rand(1) * 2 - 1) * np.pi
            y = np.matmul(calcDCM(z, th), x)

            self.assertTrue(np.abs(calcang(x, y, z) - th) < 1e-15)

    def test_projplane(self):
        """Check dot product of output vector with orthogonal direction (should be
        zero)"""

        v = np.random.rand(3, 100)
        nv = np.random.rand(3, 1)
        tmp = projplane(v, nv)
        self.assertTrue(
            np.all(
                np.abs(np.vstack([np.dot(x, nv.flatten()) * nv for x in tmp.T]).T)
                < 1e-15
            )
        )
