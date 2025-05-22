.. angutils documentation master file, created by
   sphinx-quickstart on Wed May 21 15:20:23 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

angutils
======================
``angutils`` provides routines for operating on 3D Euclidean vectors and computing a variety of angles and projections. The majority of the routines provided by this package are numerical versions of the equivalent symbolic routines provided by package ``sympyhelpers`` (https://sympyhelpers.readthedocs.io/en/latest/).

Installation
----------------

To install from PyPI: ::

    pip install angutils


To install from source, clone or download the code repository, and, in the top level directory run: ::

    pip install .


Notational Conventions
==========================
The following notational conventions are adopted throughout the code and documentation:

* All matrices are represented by uppercase unbolded letters (:math:`A`)
* All vectors are represented by lowercase bolded letters (:math:`\mathbf v`). All vectors are assumed to be Euclidean (:math:`\mathbb{R}^3`)
* Unit vectors are represented by lowercase bolded letters with hats (:math:`{\hat{\mathbf{v}}}` is the unit vector of :math:`\mathbf v`)
* Vector norms are denoted by :math:`\Vert \mathbf v \Vert` such that :math:`\mathbf v = \Vert \mathbf v \Vert {\hat{\mathbf{v}}}`
* Reference frames are represented by calligraphic capital letters (:math:`\mathcal A`) and defined by a  set of three mutually orthogonal unit vectors such that  :math:`\mathcal A = ({\hat{\mathbf{a}}}_1, {\hat{\mathbf{a}}}_2, {\hat{\mathbf{a}}}_3)` implies that :math:`{\hat{\mathbf{a}}}_1\times {\hat{\mathbf{a}}}_2 = {\hat{\mathbf{a}}_3}`. All reference frames are dextral
* The component representation of any vector :math:`\mathbf v` in reference frame :math:`\mathcal I` is given by :math:`[\mathbf v]_\mathcal I` and is assumed to be a (3x1) column matrix
* Direction cosine matrices (DCMs) are defined as :math:`{{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{A}}}` such that:

  .. math::
    [\mathbf v]_\mathcal B = {{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{A}}}[\mathbf v]_\mathcal A
* The inverse of a direction cosine matrix is its transpose and is denoted by switching the order of the superscripts: 
  
  .. math::
    \left({{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{A}}}\right)^{-1} \equiv \left({{\vphantom{C}}^{\mathcal{B}}\!{C}^{\mathcal{A}}}\right)^T = {{\vphantom{C}}^{\mathcal{A}}\!{C}^{\mathcal{B}}}

Relationship Between DCMs and Rotation Matrices
=================================================
This often causes a lot of confusion, as DCMs and 3D rotation matrices have essentially the same form and content, but serve different purposes.  A DCM maps the components of a vector expressed in one reference frame to components of the same vector expressed in a different reference frame (this is equivalent to a change of basis).  A rotation matrix operates on the components of a vector in one reference frames to return the components of a different vector in the same frame (the new vector is rotated from the original vector by the angle encoded by the rotation matrix and has the same magnitude as the original vector). 

The action of the DCM is encoded in the notational conventions, above.  The rotation matrix can best be understood by positing a dyadic (second-order tensor) :math:`\mathbb{R}` such that:

.. math::
   \mathbf{b} = \mathbb{R} \cdot \mathbf{a}

where :math:`\mathbf{b}` is :math:`\mathbf{a}`, rotated by some angle :math:`\theta` about an axis :math:`\mathbf{\hat{n}}`.  If we express :math:`\mathbb{R}` in components of some frame :math:`\mathcal{A}`, then we get a matrix whose contents match the DCM :math:`{{\vphantom{C}}^{\mathcal{A}}\!{C}^{\mathcal{B}}}`. Thus, rotation matrices and DCMs are essentially transposes of one another.

Error Quantification
=======================

As we have two methods that are effective inverses of one another (:py:meth:`~angutils.angutils.calcDCM` and :py:meth:`~angutils.angutils.DCM2axang`), we can check the numerical errors associated with conversions between axis/angle and DCM representations of the same rotations.  

|pic1|  |pic2|

    .. |pic1| image:: DCM_roundtrip_theta_errs.png
        :width: 49%

    .. |pic2| image:: DCM_roundtrip_n_errs.png
        :width: 49%

These figures show the absolute error in angle (left) and maximum absolute error in axis component when converting from an initial axis/angle representation to a DCM and then inverting the process.  As expected, errors are functions of the angle, and peak around multiples of :math:`\pi`.

The other pair of methods that serve as mutual inverses are :py:meth:`~angutils.angutils.EulerAng2DCM` and :py:meth:`~angutils.angutils.DCM2EulerAng`.  Here, however, a direct comparison is more difficult, as Euler angle sets are not guaranteed to be unique (e.g., there is a :math:`\pi` radian ambiguity in all terms of an Euler angle set associated with a specific rotation order.  Instead of directly comparing angle inputs to :py:meth:`~angutils.angutils.EulerAng2DCM` to the angle outputs of :py:meth:`~angutils.angutils.DCM2EulerAng`, we can instead use those outputs to re-compute the DCM, and then compare the entries of the two DCMs.  Carrying out this exercise (which is implemented in the unit tests for the package) shows that all pairs of DCM entries produced in this fashion have a guaranteed maximum error of 1e-15. 

.. toctree::
   :maxdepth: 4
   :caption: API:

   angutils

