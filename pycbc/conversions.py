# Copyright (C) 2017  Collin Capano, Christopher M. Biwer, Duncan Brown,
# and Steven Reyes
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#
"""
This module provides a library of functions that calculate waveform parameters
from other parameters. All exposed functions in this module's namespace return
one parameter given a set of inputs.
"""

import copy
import numpy
import lal
from pycbc.detector import Detector
import pycbc.cosmology
from .coordinates import (
    spherical_to_cartesian as _spherical_to_cartesian,
    cartesian_to_spherical as _cartesian_to_spherical)
from pycbc import neutron_stars as ns

pykerr = pycbc.libutils.import_optional('pykerr')
lalsim = pycbc.libutils.import_optional('lalsimulation')

#
# =============================================================================
#
#                           Helper functions
#
# =============================================================================
#
def ensurearray(*args):
    """Apply numpy's broadcast rules to the given arguments.

    This will ensure that all of the arguments are numpy arrays and that they
    all have the same shape. See ``numpy.broadcast_arrays`` for more details.

    It also returns a boolean indicating whether any of the inputs were
    originally arrays.

    Parameters
    ----------
    *args :
        The arguments to check.

    Returns
    -------
    list :
        A list with length ``N+1`` where ``N`` is the number of given
        arguments. The first N values are the input arguments as ``ndarrays``s.
        The last value is a boolean indicating whether any of the
        inputs was an array.
    """
    input_is_array = any(isinstance(arg, numpy.ndarray) for arg in args)
    args = numpy.broadcast_arrays(*args)
    args.append(input_is_array)
    return args


def formatreturn(arg, input_is_array=False):
    """If the given argument is a numpy array with shape (1,), just returns
    that value."""
    if not input_is_array and arg.size == 1:
        arg = arg.item()
    return arg

#
# =============================================================================
#
#                           Fundamental conversions
#
# =============================================================================
#

def sec_to_year(sec):
    """ Converts number of seconds to number of years """
    return sec / lal.YRJUL_SI


#
# =============================================================================
#
#                           CBC mass functions
#
# =============================================================================
#
def primary_mass(mass1, mass2):
    """Returns the larger of mass1 and mass2 (p = primary)."""
    mass1, mass2, input_is_array = ensurearray(mass1, mass2)
    mp = copy.copy(mass1)
    mask = mass1 < mass2
    mp[mask] = mass2[mask]
    return formatreturn(mp, input_is_array)


def secondary_mass(mass1, mass2):
    """Returns the smaller of mass1 and mass2 (s = secondary)."""
    mass1, mass2, input_is_array = ensurearray(mass1, mass2)
    if mass1.shape != mass2.shape:
        raise ValueError("mass1 and mass2 must have same shape")
    ms = copy.copy(mass2)
    mask = mass1 < mass2
    ms[mask] = mass1[mask]
    return formatreturn(ms, input_is_array)


def mtotal_from_mass1_mass2(mass1, mass2):
    """Returns the total mass from mass1 and mass2."""
    return mass1 + mass2


def q_from_mass1_mass2(mass1, mass2):
    """Returns the mass ratio m1/m2, where m1 >= m2."""
    return primary_mass(mass1, mass2) / secondary_mass(mass1, mass2)


def invq_from_mass1_mass2(mass1, mass2):
    """Returns the inverse mass ratio m2/m1, where m1 >= m2."""
    return secondary_mass(mass1, mass2) / primary_mass(mass1, mass2)


def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1*mass2 / (mass1 + mass2)**2.


def mchirp_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1 + mass2)


def mass1_from_mtotal_q(mtotal, q):
    """Returns a component mass from the given total mass and mass ratio.

    If the mass ratio q is >= 1, the returned mass will be the primary
    (heavier) mass. If q < 1, the returned mass will be the secondary
    (lighter) mass.
    """
    return q*mtotal / (1. + q)


def mass2_from_mtotal_q(mtotal, q):
    """Returns a component mass from the given total mass and mass ratio.

    If the mass ratio q is >= 1, the returned mass will be the secondary
    (lighter) mass. If q < 1, the returned mass will be the primary (heavier)
    mass.
    """
    return mtotal / (1. + q)


def mass1_from_mtotal_eta(mtotal, eta):
    """Returns the primary mass from the total mass and symmetric mass
    ratio.
    """
    return 0.5 * mtotal * (1.0 + (1.0 - 4.0 * eta)**0.5)


def mass2_from_mtotal_eta(mtotal, eta):
    """Returns the secondary mass from the total mass and symmetric mass
    ratio.
    """
    return 0.5 * mtotal * (1.0 - (1.0 - 4.0 * eta)**0.5)


def mtotal_from_mchirp_eta(mchirp, eta):
    """Returns the total mass from the chirp mass and symmetric mass ratio.
    """
    return mchirp / eta**(3./5.)


def mass1_from_mchirp_eta(mchirp, eta):
    """Returns the primary mass from the chirp mass and symmetric mass ratio.
    """
    mtotal = mtotal_from_mchirp_eta(mchirp, eta)
    return mass1_from_mtotal_eta(mtotal, eta)


def mass2_from_mchirp_eta(mchirp, eta):
    """Returns the primary mass from the chirp mass and symmetric mass ratio.
    """
    mtotal = mtotal_from_mchirp_eta(mchirp, eta)
    return mass2_from_mtotal_eta(mtotal, eta)


def _mass2_from_mchirp_mass1(mchirp, mass1):
    r"""Returns the secondary mass from the chirp mass and primary mass.

    As this is a cubic equation this requires finding the roots and returning
    the one that is real. Basically it can be shown that:

    .. math::
        m_2^3 - a(m_2 + m_1) = 0,

    where

    .. math::
        a = \frac{\mathcal{M}^5}{m_1^3}.

    This has 3 solutions but only one will be real.
    """
    a = mchirp**5 / mass1**3
    roots = numpy.roots([1, 0, -a, -a * mass1])
    # Find the real one
    real_root = roots[(abs(roots - roots.real)).argmin()]
    return real_root.real

mass2_from_mchirp_mass1 = numpy.vectorize(_mass2_from_mchirp_mass1)


def _mass_from_knownmass_eta(known_mass, eta, known_is_secondary=False,
                            force_real=True):
    r"""Returns the other component mass given one of the component masses
    and the symmetric mass ratio.

    This requires finding the roots of the quadratic equation:

    .. math::
        \eta m_2^2 + (2\eta - 1)m_1 m_2 + \eta m_1^2 = 0.

    This has two solutions which correspond to :math:`m_1` being the heavier
    mass or it being the lighter mass. By default, `known_mass` is assumed to
    be the heavier (primary) mass, and the smaller solution is returned. Use
    the `other_is_secondary` to invert.

    Parameters
    ----------
    known_mass : float
        The known component mass.
    eta : float
        The symmetric mass ratio.
    known_is_secondary : {False, bool}
        Whether the known component mass is the primary or the secondary. If
        True, `known_mass` is assumed to be the secondary (lighter) mass and
        the larger solution is returned. Otherwise, the smaller solution is
        returned. Default is False.
    force_real : {True, bool}
        Force the returned mass to be real.

    Returns
    -------
    float
        The other component mass.
    """
    roots = numpy.roots([eta, (2*eta - 1) * known_mass, eta * known_mass**2.])
    if force_real:
        roots = numpy.real(roots)
    if known_is_secondary:
        return roots[roots.argmax()]
    else:
        return roots[roots.argmin()]

mass_from_knownmass_eta = numpy.vectorize(_mass_from_knownmass_eta)


def mass2_from_mass1_eta(mass1, eta, force_real=True):
    """Returns the secondary mass from the primary mass and symmetric mass
    ratio.
    """
    return mass_from_knownmass_eta(mass1, eta, known_is_secondary=False,
                                   force_real=force_real)


def mass1_from_mass2_eta(mass2, eta, force_real=True):
    """Returns the primary mass from the secondary mass and symmetric mass
    ratio.
    """
    return mass_from_knownmass_eta(mass2, eta, known_is_secondary=True,
                                   force_real=force_real)


def eta_from_q(q):
    r"""Returns the symmetric mass ratio from the given mass ratio.

    This is given by:

    .. math::
        \eta = \frac{q}{(1+q)^2}.

    Note that the mass ratio may be either < 1 or > 1.
    """
    return q / (1. + q)**2


def mass1_from_mchirp_q(mchirp, q):
    """Returns the primary mass from the given chirp mass and mass ratio."""
    mass1 = q**(2./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass1


def mass2_from_mchirp_q(mchirp, q):
    """Returns the secondary mass from the given chirp mass and mass ratio."""
    mass2 = q**(-3./5.) * (1.0 + q)**(1./5.) * mchirp
    return mass2


def _a0(f_lower):
    """Used in calculating chirp times: see Cokelaer, arxiv.org:0706.4437
       appendix 1, also lalinspiral/python/sbank/tau0tau3.py.
    """
    return 5. / (256. * (numpy.pi * f_lower)**(8./3.))


def _a3(f_lower):
    """Another parameter used for chirp times"""
    return numpy.pi / (8. * (numpy.pi * f_lower)**(5./3.))


def tau0_from_mtotal_eta(mtotal, eta, f_lower):
    r"""Returns :math:`\tau_0` from the total mass, symmetric mass ratio, and
    the given frequency.
    """
    # convert to seconds
    mtotal = mtotal * lal.MTSUN_SI
    # formulae from arxiv.org:0706.4437
    return _a0(f_lower) / (mtotal**(5./3.) * eta)


def tau0_from_mchirp(mchirp, f_lower):
    r"""Returns :math:`\tau_0` from the chirp mass and the given frequency.
    """
    # convert to seconds
    mchirp = mchirp * lal.MTSUN_SI
    # formulae from arxiv.org:0706.4437
    return _a0(f_lower) / mchirp ** (5./3.)


def tau3_from_mtotal_eta(mtotal, eta, f_lower):
    r"""Returns :math:`\tau_0` from the total mass, symmetric mass ratio, and
    the given frequency.
    """
    # convert to seconds
    mtotal = mtotal * lal.MTSUN_SI
    # formulae from arxiv.org:0706.4437
    return _a3(f_lower) / (mtotal**(2./3.) * eta)


def tau0_from_mass1_mass2(mass1, mass2, f_lower):
    r"""Returns :math:`\tau_0` from the component masses and given frequency.
    """
    mtotal = mass1 + mass2
    eta = eta_from_mass1_mass2(mass1, mass2)
    return tau0_from_mtotal_eta(mtotal, eta, f_lower)


def tau3_from_mass1_mass2(mass1, mass2, f_lower):
    r"""Returns :math:`\tau_3` from the component masses and given frequency.
    """
    mtotal = mass1 + mass2
    eta = eta_from_mass1_mass2(mass1, mass2)
    return tau3_from_mtotal_eta(mtotal, eta, f_lower)


def mchirp_from_tau0(tau0, f_lower):
    r"""Returns chirp mass from :math:`\tau_0` and the given frequency.
    """
    mchirp = (_a0(f_lower) / tau0) ** (3./5.)  # in seconds
    # convert back to solar mass units
    return mchirp / lal.MTSUN_SI


def mtotal_from_tau0_tau3(tau0, tau3, f_lower,
                          in_seconds=False):
    r"""Returns total mass from :math:`\tau_0, \tau_3`."""
    mtotal = (tau3 / _a3(f_lower)) / (tau0 / _a0(f_lower))
    if not in_seconds:
        # convert back to solar mass units
        mtotal /= lal.MTSUN_SI
    return mtotal


def eta_from_tau0_tau3(tau0, tau3, f_lower):
    r"""Returns symmetric mass ratio from :math:`\tau_0, \tau_3`."""
    mtotal = mtotal_from_tau0_tau3(tau0, tau3, f_lower,
                                   in_seconds=True)
    eta = mtotal**(-2./3.) * (_a3(f_lower) / tau3)
    return eta


def mass1_from_tau0_tau3(tau0, tau3, f_lower):
    r"""Returns the primary mass from the given :math:`\tau_0, \tau_3`."""
    mtotal = mtotal_from_tau0_tau3(tau0, tau3, f_lower)
    eta = eta_from_tau0_tau3(tau0, tau3, f_lower)
    return mass1_from_mtotal_eta(mtotal, eta)


def mass2_from_tau0_tau3(tau0, tau3, f_lower):
    r"""Returns the secondary mass from the given :math:`\tau_0, \tau_3`."""
    mtotal = mtotal_from_tau0_tau3(tau0, tau3, f_lower)
    eta = eta_from_tau0_tau3(tau0, tau3, f_lower)
    return mass2_from_mtotal_eta(mtotal, eta)


def lambda_tilde(mass1, mass2, lambda1, lambda2):
    """ The effective lambda parameter

    The mass-weighted dominant effective lambda parameter defined in
    https://journals.aps.org/prd/pdf/10.1103/PhysRevD.91.043002
    """
    m1, m2, lambda1, lambda2, input_is_array = ensurearray(
        mass1, mass2, lambda1, lambda2)
    lsum = lambda1 + lambda2
    ldiff, _ = ensurearray(lambda1 - lambda2)
    mask = m1 < m2
    ldiff[mask] = -ldiff[mask]
    eta = eta_from_mass1_mass2(m1, m2)
    p1 = (lsum) * (1 + 7. * eta - 31 * eta ** 2.0)
    p2 = (1 - 4 * eta)**0.5 * (1 + 9 * eta - 11 * eta ** 2.0) * (ldiff)
    return formatreturn(8.0 / 13.0 * (p1 + p2), input_is_array)


def lambda_from_mass_tov_file(mass, tov_file, distance=0.):
    """Return the lambda parameter(s) corresponding to the input mass(es)
    interpolating from the mass-Lambda data for a particular EOS read in from
    an ASCII file.
    """
    data = numpy.loadtxt(tov_file)
    mass_from_file = data[:, 0]
    lambda_from_file = data[:, 1]
    mass_src = mass/(1.0 + pycbc.cosmology.redshift(distance))
    lambdav = numpy.interp(mass_src, mass_from_file, lambda_from_file)
    return lambdav


def remnant_mass_from_mass1_mass2_spherical_spin_eos(
        mass1, mass2, spin1a=0.0, spin1pol=0.0, eos='2H'):
    """
    Function that determines the remnant disk mass of an NS-BH system
    using the fit to numerical-relativity results discussed in
    Foucart, Hinderer & Nissanke, PRD 98, 081501(R) (2018).
    The BH spin may be misaligned with the orbital angular momentum.
    In such cases the ISSO is approximated following the approach of
    Stone, Loeb & Berger, PRD 87, 084053 (2013), which was originally
    devised for a previous NS-BH remnant mass fit of
    Foucart, PRD 86, 124007 (2012).
    Note: NS spin is assumed to be 0!

    Parameters
    -----------
    mass1 : float
        The mass of the black hole, in solar masses.
    mass2 : float
        The mass of the neutron star, in solar masses.
    spin1a : float, optional
        The dimensionless magnitude of the spin of mass1. Default = 0.
    spin1pol : float, optional
        The tilt angle of the spin of mass1. Default = 0 (aligned w L).
    eos : str, optional
        Name of the equation of state being adopted. Default is '2H'.

    Returns
    ----------
    remnant_mass: float
        The remnant mass in solar masses
    """
    mass1, mass2, spin1a, spin1pol, input_is_array = ensurearray(
        mass1, mass2, spin1a, spin1pol)
    # mass1 must be greater than mass2
    try:
        if any(mass2 > mass1) and input_is_array:
            raise ValueError(f'Require mass1 >= mass2')
    except TypeError:
        if mass2 > mass1 and not input_is_array:
            raise ValueError(f'Require mass1 >= mass2. {mass1} < {mass2}')
    ns_compactness, ns_b_mass = ns.initialize_eos(mass2, eos)
    eta = eta_from_mass1_mass2(mass1, mass2)
    remnant_mass = ns.foucart18(
        eta, ns_compactness, ns_b_mass, spin1a, spin1pol)
    return formatreturn(remnant_mass, input_is_array)


def remnant_mass_from_mass1_mass2_cartesian_spin_eos(
        mass1, mass2, spin1x=0.0, spin1y=0.0, spin1z=0.0, eos='2H'):
    """
    Function that determines the remnant disk mass of an NS-BH system
    using the fit to numerical-relativity results discussed in
    Foucart, Hinderer & Nissanke, PRD 98, 081501(R) (2018).
    The BH spin may be misaligned with the orbital angular momentum.
    In such cases the ISSO is approximated following the approach of
    Stone, Loeb & Berger, PRD 87, 084053 (2013), which was originally
    devised for a previous NS-BH remnant mass fit of
    Foucart, PRD 86, 124007 (2012).
    Note: NS spin is assumed to be 0!

    Parameters
    -----------
    mass1 : float
        The mass of the black hole, in solar masses.
    mass2 : float
        The mass of the neutron star, in solar masses.
    spin1x : float, optional
        The dimensionless x-component of the spin of mass1. Default = 0.
    spin1y : float, optional
        The dimensionless y-component of the spin of mass1. Default = 0.
    spin1z : float, optional
        The dimensionless z-component of the spin of mass1. Default = 0.
    eos: str, optional
        Name of the equation of state being adopted. Default is '2H'.

    Returns
    ----------
    remnant_mass: float
        The remnant mass in solar masses
    """
    spin1a, _, spin1pol = _cartesian_to_spherical(spin1x, spin1y, spin1z)
    return remnant_mass_from_mass1_mass2_spherical_spin_eos(
        mass1, mass2, spin1a, spin1pol, eos=eos)


#
# =============================================================================
#
#                           CBC spin functions
#
# =============================================================================
#
def chi_eff(mass1, mass2, spin1z, spin2z):
    """Returns the effective spin from mass1, mass2, spin1z, and spin2z."""
    return (spin1z * mass1 + spin2z * mass2) / (mass1 + mass2)


def chi_a(mass1, mass2, spin1z, spin2z):
    """ Returns the aligned mass-weighted spin difference from mass1, mass2,
    spin1z, and spin2z.
    """
    return (spin2z * mass2 - spin1z * mass1) / (mass2 + mass1)


def chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Returns the effective precession spin from mass1, mass2, spin1x,
    spin1y, spin2x, and spin2y.
    """
    xi1 = secondary_xi(mass1, mass2, spin1x, spin1y, spin2x, spin2y)
    xi2 = primary_xi(mass1, mass2, spin1x, spin1y, spin2x, spin2y)
    return chi_p_from_xi1_xi2(xi1, xi2)


def phi_a(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """ Returns the angle between the in-plane perpendicular spins."""
    phi1 = phi_from_spinx_spiny(primary_spin(mass1, mass2, spin1x, spin2x),
                                primary_spin(mass1, mass2, spin1y, spin2y))
    phi2 = phi_from_spinx_spiny(secondary_spin(mass1, mass2, spin1x, spin2x),
                                secondary_spin(mass1, mass2, spin1y, spin2y))
    return (phi1 - phi2) % (2 * numpy.pi)


def phi_s(spin1x, spin1y, spin2x, spin2y):
    """ Returns the sum of the in-plane perpendicular spins."""
    phi1 = phi_from_spinx_spiny(spin1x, spin1y)
    phi2 = phi_from_spinx_spiny(spin2x, spin2y)
    return (phi1 + phi2) % (2 * numpy.pi)


def chi_eff_from_spherical(mass1, mass2, spin1_a, spin1_polar,
                           spin2_a, spin2_polar):
    """Returns the effective spin using spins in spherical coordinates."""
    spin1z = spin1_a * numpy.cos(spin1_polar)
    spin2z = spin2_a * numpy.cos(spin2_polar)
    return chi_eff(mass1, mass2, spin1z, spin2z)


def chi_p_from_spherical(mass1, mass2, spin1_a, spin1_azimuthal, spin1_polar,
                         spin2_a, spin2_azimuthal, spin2_polar):
    """Returns the effective precession spin using spins in spherical
    coordinates.
    """
    spin1x, spin1y, _ = _spherical_to_cartesian(
        spin1_a, spin1_azimuthal, spin1_polar)
    spin2x, spin2y, _ = _spherical_to_cartesian(
        spin2_a, spin2_azimuthal, spin2_polar)
    return chi_p(mass1, mass2, spin1x, spin1y, spin2x, spin2y)


def primary_spin(mass1, mass2, spin1, spin2):
    """Returns the dimensionless spin of the primary mass."""
    mass1, mass2, spin1, spin2, input_is_array = ensurearray(
        mass1, mass2, spin1, spin2)
    sp = copy.copy(spin1)
    mask = mass1 < mass2
    sp[mask] = spin2[mask]
    return formatreturn(sp, input_is_array)


def secondary_spin(mass1, mass2, spin1, spin2):
    """Returns the dimensionless spin of the secondary mass."""
    mass1, mass2, spin1, spin2, input_is_array = ensurearray(
        mass1, mass2, spin1, spin2)
    ss = copy.copy(spin2)
    mask = mass1 < mass2
    ss[mask] = spin1[mask]
    return formatreturn(ss, input_is_array)


def primary_xi(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Returns the effective precession spin argument for the larger mass.
    """
    spinx = primary_spin(mass1, mass2, spin1x, spin2x)
    spiny = primary_spin(mass1, mass2, spin1y, spin2y)
    return chi_perp_from_spinx_spiny(spinx, spiny)


def secondary_xi(mass1, mass2, spin1x, spin1y, spin2x, spin2y):
    """Returns the effective precession spin argument for the smaller mass.
    """
    spinx = secondary_spin(mass1, mass2, spin1x, spin2x)
    spiny = secondary_spin(mass1, mass2, spin1y, spin2y)
    return xi2_from_mass1_mass2_spin2x_spin2y(mass1, mass2, spinx, spiny)


def xi1_from_spin1x_spin1y(spin1x, spin1y):
    """Returns the effective precession spin argument for the larger mass.
    This function assumes it's given spins of the primary mass.
    """
    return chi_perp_from_spinx_spiny(spin1x, spin1y)


def xi2_from_mass1_mass2_spin2x_spin2y(mass1, mass2, spin2x, spin2y):
    """Returns the effective precession spin argument for the smaller mass.
    This function assumes it's given spins of the secondary mass.
    """
    q = q_from_mass1_mass2(mass1, mass2)
    a1 = 2 + 3 * q / 2
    a2 = 2 + 3 / (2 * q)
    return a1 / (q**2 * a2) * chi_perp_from_spinx_spiny(spin2x, spin2y)


def chi_perp_from_spinx_spiny(spinx, spiny):
    """Returns the in-plane spin from the x/y components of the spin.
    """
    return numpy.sqrt(spinx**2 + spiny**2)


def chi_perp_from_mass1_mass2_xi2(mass1, mass2, xi2):
    """Returns the in-plane spin from mass1, mass2, and xi2 for the
    secondary mass.
    """
    q = q_from_mass1_mass2(mass1, mass2)
    a1 = 2 + 3 * q / 2
    a2 = 2 + 3 / (2 * q)
    return q**2 * a2 / a1 * xi2


def chi_p_from_xi1_xi2(xi1, xi2):
    """Returns effective precession spin from xi1 and xi2.
    """
    xi1, xi2, input_is_array = ensurearray(xi1, xi2)
    chi_p = copy.copy(xi1)
    mask = xi1 < xi2
    chi_p[mask] = xi2[mask]
    return formatreturn(chi_p, input_is_array)


def phi1_from_phi_a_phi_s(phi_a, phi_s):
    """Returns the angle between the x-component axis and the in-plane
    spin for the primary mass from phi_s and phi_a.
    """
    return (phi_s + phi_a) / 2.0


def phi2_from_phi_a_phi_s(phi_a, phi_s):
    """Returns the angle between the x-component axis and the in-plane
    spin for the secondary mass from phi_s and phi_a.
    """
    return (phi_s - phi_a) / 2.0


def phi_from_spinx_spiny(spinx, spiny):
    """Returns the angle between the x-component axis and the in-plane spin.
    """
    phi = numpy.arctan2(spiny, spinx)
    return phi % (2 * numpy.pi)


def spin1z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2, chi_eff, chi_a):
    """Returns spin1z.
    """
    return (mass1 + mass2) / (2.0 * mass1) * (chi_eff - chi_a)


def spin2z_from_mass1_mass2_chi_eff_chi_a(mass1, mass2, chi_eff, chi_a):
    """Returns spin2z.
    """
    return (mass1 + mass2) / (2.0 * mass2) * (chi_eff + chi_a)


def spin1x_from_xi1_phi_a_phi_s(xi1, phi_a, phi_s):
    """Returns x-component spin for primary mass.
    """
    phi1 = phi1_from_phi_a_phi_s(phi_a, phi_s)
    return xi1 * numpy.cos(phi1)


def spin1y_from_xi1_phi_a_phi_s(xi1, phi_a, phi_s):
    """Returns y-component spin for primary mass.
    """
    phi1 = phi1_from_phi_a_phi_s(phi_s, phi_a)
    return xi1 * numpy.sin(phi1)


def spin2x_from_mass1_mass2_xi2_phi_a_phi_s(mass1, mass2, xi2, phi_a, phi_s):
    """Returns x-component spin for secondary mass.
    """
    chi_perp = chi_perp_from_mass1_mass2_xi2(mass1, mass2, xi2)
    phi2 = phi2_from_phi_a_phi_s(phi_a, phi_s)
    return chi_perp * numpy.cos(phi2)


def spin2y_from_mass1_mass2_xi2_phi_a_phi_s(mass1, mass2, xi2, phi_a, phi_s):
    """Returns y-component spin for secondary mass.
    """
    chi_perp = chi_perp_from_mass1_mass2_xi2(mass1, mass2, xi2)
    phi2 = phi2_from_phi_a_phi_s(phi_a, phi_s)
    return chi_perp * numpy.sin(phi2)


def dquadmon_from_lambda(lambdav):
    r"""Return the quadrupole moment of a neutron star given its lambda

    We use the relations defined here. https://arxiv.org/pdf/1302.4499.pdf.
    Note that the convention we use is that:

    .. math::

        \mathrm{dquadmon} = \bar{Q} - 1.

    Where :math:`\bar{Q}` (dimensionless) is the reduced quadrupole moment.
    """
    ll = numpy.log(lambdav)
    ai = .194
    bi = .0936
    ci = 0.0474
    di = -4.21 * 10**-3.0
    ei = 1.23 * 10**-4.0
    ln_quad_moment = ai + bi*ll + ci*ll**2.0 + di*ll**3.0 + ei*ll**4.0
    return numpy.exp(ln_quad_moment) - 1


def spin_from_pulsar_freq(mass, radius, freq):
    """Returns the dimensionless spin of a pulsar.

    Assumes the pulsar is a solid sphere when computing the moment of inertia.

    Parameters
    ----------
    mass : float
        The mass of the pulsar, in solar masses.
    radius : float
        The assumed radius of the pulsar, in kilometers.
    freq : float
        The spin frequency of the pulsar, in Hz.
    """
    omega = 2 * numpy.pi * freq
    mt = mass * lal.MTSUN_SI
    mominert = (2/5.) * mt * (radius * 1000 / lal.C_SI)**2
    return mominert * omega / mt**2


#
# =============================================================================
#
#                         Extrinsic parameter functions
#
# =============================================================================
#
def chirp_distance(dist, mchirp, ref_mass=1.4):
    """Returns the chirp distance given the luminosity distance and chirp mass.
    """
    return dist * (2.**(-1./5) * ref_mass / mchirp)**(5./6)


def distance_from_chirp_distance_mchirp(chirp_distance, mchirp, ref_mass=1.4):
    """Returns the luminosity distance given a chirp distance and chirp mass.
    """
    return chirp_distance * (2.**(-1./5) * ref_mass / mchirp)**(-5./6)


_detector_cache = {}
def det_tc(detector_name, ra, dec, tc, ref_frame='geocentric', relative=False):
    """Returns the coalescence time of a signal in the given detector.

    Parameters
    ----------
    detector_name : string
        The name of the detector, e.g., 'H1'.
    ra : float
        The right ascension of the signal, in radians.
    dec : float
        The declination of the signal, in radians.
    tc : float
        The GPS time of the coalescence of the signal in the `ref_frame`.
    ref_frame : {'geocentric', string}
        The reference frame that the given coalescence time is defined in.
        May specify 'geocentric', or a detector name; default is 'geocentric'.

    Returns
    -------
    float :
        The GPS time of the coalescence in detector `detector_name`.
    """
    ref_time = tc
    if relative:
        tc = 0

    if ref_frame == detector_name:
        return tc
    if detector_name not in _detector_cache:
        _detector_cache[detector_name] = Detector(detector_name)
    detector = _detector_cache[detector_name]
    if ref_frame == 'geocentric':
        return tc + detector.time_delay_from_earth_center(ra, dec, ref_time)
    else:
        other = Detector(ref_frame)
        return tc + detector.time_delay_from_detector(other, ra, dec, ref_time)

def optimal_orientation_from_detector(detector_name, tc):
    """ Low-level function to be called from _optimal_dec_from_detector
    and _optimal_ra_from_detector"""

    d = Detector(detector_name)
    ra, dec = d.optimal_orientation(tc)
    return ra, dec

def optimal_dec_from_detector(detector_name, tc):
    """For a given detector and GPS time, return the optimal orientation
    (directly overhead of the detector) in declination.


    Parameters
    ----------
    detector_name : string
        The name of the detector, e.g., 'H1'.
    tc : float
        The GPS time of the coalescence of the signal in the `ref_frame`.

    Returns
    -------
    float :
        The declination of the signal, in radians.
    """
    return optimal_orientation_from_detector(detector_name, tc)[1]

def optimal_ra_from_detector(detector_name, tc):
    """For a given detector and GPS time, return the optimal orientation
    (directly overhead of the detector) in right ascension.


    Parameters
    ----------
    detector_name : string
        The name of the detector, e.g., 'H1'.
    tc : float
        The GPS time of the coalescence of the signal in the `ref_frame`.

    Returns
    -------
    float :
        The declination of the signal, in radians.
    """
    return optimal_orientation_from_detector(detector_name, tc)[0]

#
# =============================================================================
#
#                         Likelihood statistic parameter functions
#
# =============================================================================
#
def snr_from_loglr(loglr):
    """Returns SNR computed from the given log likelihood ratio(s). This is
    defined as `sqrt(2*loglr)`.If the log likelihood ratio is < 0, returns 0.

    Parameters
    ----------
    loglr : array or float
        The log likelihood ratio(s) to evaluate.

    Returns
    -------
    array or float
        The SNRs computed from the log likelihood ratios.
    """
    singleval = isinstance(loglr, float)
    if singleval:
        loglr = numpy.array([loglr])
    # temporarily quiet sqrt(-1) warnings
    with numpy.errstate(invalid="ignore"):
        snrs = numpy.sqrt(2*loglr)
    snrs[numpy.isnan(snrs)] = 0.
    if singleval:
        snrs = snrs[0]
    return snrs

#
# =============================================================================
#
#                         BH Ringdown functions
#
# =============================================================================
#


def get_lm_f0tau(mass, spin, l, m, n=0, which='both'):
    """Return the f0 and the tau for one or more overtones of an l, m mode.

    Parameters
    ----------
    mass : float or array
        Mass of the black hole (in solar masses).
    spin : float or array
        Dimensionless spin of the final black hole.
    l : int or array
        l-index of the harmonic.
    m : int or array
        m-index of the harmonic.
    n : int or array
        Overtone(s) to generate, where n=0 is the fundamental mode.
        Default is 0.
    which : {'both', 'f0', 'tau'}, optional
        What to return; 'both' returns both frequency and tau, 'f0' just
        frequency, 'tau' just tau. Default is 'both'.

    Returns
    -------
    f0 : float or array
        Returned if ``which`` is 'both' or 'f0'.
        The frequency of the QNM(s), in Hz.
    tau : float or array
        Returned if ``which`` is 'both' or 'tau'.
        The damping time of the QNM(s), in seconds.
    """
    # convert to arrays
    mass, spin, l, m, n, input_is_array = ensurearray(
        mass, spin, l, m, n)
    # we'll ravel the arrays so we can evaluate each parameter combination
    # one at a a time
    getf0 = which == 'both' or which == 'f0'
    gettau = which == 'both' or which == 'tau'
    out = []
    if getf0:
        f0s = pykerr.qnmfreq(mass, spin, l, m, n)
        out.append(formatreturn(f0s, input_is_array))
    if gettau:
        taus = pykerr.qnmtau(mass, spin, l, m, n)
        out.append(formatreturn(taus, input_is_array))
    if not (getf0 and gettau):
        out = out[0]
    return out

####################### JP Ringdown Functions start #####################
def func(y,spin,epsilon):
    com_term = common(y,spin,epsilon)# (9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 
    if com_term !=None:
        unc = y**2* ( (9*epsilon**3*spin**2*y - 30*epsilon**3*spin**2 + 9*epsilon**2*spin**2*y**4 - 42*epsilon**2*spin**2*y**3 \
                       - 4*epsilon**2*spin*y**4*numpy.sqrt(com_term)
                       + 9*epsilon**2*y**6 - 48*epsilon**2*y**5 + 64*epsilon**2*y**4 - 12*epsilon*spin**2*y**6 
                       - 8*epsilon*spin*y**7*numpy.sqrt(com_term) 
                       - epsilon*y**9*(com_term)
                       + 2*epsilon*y**8*(com_term) 
                       - 12*epsilon*y**8 + 32*epsilon*y**7 
                       - 4*spin*y**10*numpy.sqrt(com_term) 
                       - y**12 * (com_term) 
                       + 2*y**11 * (com_term) 
                       + 4*y**10 
                      )) 
        return unc 
#     print('com_term=',com_term,y,spin,epsilon) 
    return None 

def func_der(y,spin,epsilon): 
    com_term = common(y,spin,epsilon) 
    if com_term != None: 
        dunk =y*(9*epsilon**3*spin**2*y - 48*epsilon**3*spin**2 + 36*epsilon**2*spin**2*y**4 - 150*epsilon**2*spin**2*y**3 \
                   - 16*epsilon**2*spin*y**4*numpy.sqrt(com_term) 
                   + 54*epsilon**2*y**6 - 288*epsilon**2*y**5 + 384*epsilon**2*y**4 - 48*epsilon*spin**2*y**6  
                   - 32*epsilon*spin*y**7*numpy.sqrt(com_term) 
                   - epsilon*y**9 * (com_term) 
                   - 72*epsilon*y**8 + 192*epsilon*y**7  
                   - 16*spin*y**10*numpy.sqrt(com_term)  
                   - 4*y**12*(com_term)  
                   + 6*y**11*(com_term)  
                   +24*y**10 
                  ) 
        return dunk 
    return None 

def common(y,spin,epsilon): 
    C = (9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y

    if C>=0:
        return C
    return None

def IterativeFunc(ran_mx,spin,epsilon, prec):
    for i in range(len(ran_mx) -1):
        d = common(ran_mx[i],spin,epsilon)
        p = common(ran_mx[i+1],spin,epsilon)
        if d == None or p == None:
            return None
        else:
            if(func(ran_mx[i],spin,epsilon)*func(ran_mx[i+1],spin,epsilon) < 0.):
                if (((ran_mx[i+1] - ran_mx[i])/2.) <= prec):
                    b = ((ran_mx[i+1] + ran_mx[i])/2.)
                    return b

                else:
                    ran_newx = numpy.linspace(ran_mx[i],ran_mx[i+1], 1000)
                    ran_newmx = 0.5*(ran_newx[1::] + ran_newx[:-1:])
                    return IterativeFunc(ran_newmx,spin,epsilon, prec)
    return None



def LR_pos(spin,epsilon):
    prec = 1e-8
    if -10.< epsilon < -1.:
        ran_x = numpy.linspace(2.,4.3, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
    
    elif -1. <= epsilon <= 65.:
        ran_x = numpy.linspace(1.5,4., 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)

    elif 65. < epsilon <= 80.:
        ran_x = numpy.linspace(1.5,3.4, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
        
    elif 80. < epsilon <= 100.:
        ran_x = numpy.linspace(1.75,3.4, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
        
    else:
        ran_x = numpy.linspace(3.,5.5, 1000)
        ran_mx = 0.5*(ran_x[1::] + ran_x[:-1:])
        Rpos = IterativeFunc(ran_mx,spin,epsilon,prec=prec)
        
    return Rpos

def freqM_dim_less(spin,epsilon): 
    y = LR_pos(spin,epsilon)
    if y != None:
        com_term = common(y,spin,epsilon)#(9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 

    #     if com_term != None: 
        frequ = 2*((-3*epsilon*y + 8*epsilon + 2*y**3)/(8*epsilon*spin + 2*spin*y**3 + y**5*numpy.sqrt(com_term)))
        
        return frequ 
    return 10


def gamma0(spin,epsilon): 
    y = LR_pos(spin,epsilon)
    if y != None:
        com_term = common(y,spin,epsilon)#(9*epsilon**2*spin**2)/y**8 - 6*epsilon/y**3 + 16*epsilon/y**4 + 4/y 
    #     y = LR_pos(spin,epsilon)
    #     if com_term != None: 
        numerator = (54*epsilon**5*spin**6*y - 108*epsilon**5*spin**6 + 108*epsilon**4*spin**6*y**4- 270*epsilon**4*spin**6*y**3\
                    + 24*epsilon**4*spin**5*y**4*numpy.sqrt(com_term) 
                    + 117*epsilon**4*spin**4*y**6 - 480*epsilon**4*spin**4*y**5  
                    + 496*epsilon**4*spin**4*y**4 + 54*epsilon**3*spin**6*y**7 - 216*epsilon**3*spin**6*y**6  
                    + 36*epsilon**3*spin**5*y**7*numpy.sqrt(com_term) 
                    -  6*epsilon**3*spin**4*y**9*(com_term) + 126*epsilon**3*spin**4*y**9  
                    + 20*epsilon**3*spin**4*y**8*(com_term) 
                    - 648*epsilon**3*spin**4*y**8 + 808*epsilon**3*spin**4*y**7  
                    + 48*epsilon**3*spin**3*y**9*numpy.sqrt(com_term)  
                    - 96*epsilon**3*spin**3*y**8*numpy.sqrt(com_term) 
                    + 72*epsilon**3*spin**2*y**11 - 456*epsilon**3*spin**2*y**10 + 968*epsilon**3*spin**2*y**9  
                    - 688*epsilon**3*spin**2*y**8 - 54*epsilon**2*spin**6*y**9  
                    - 12*epsilon**2*spin**4*y**12*(com_term)+ 9*epsilon**2*spin**4*y**12 
                    + 42*epsilon**2*spin**4*y**11*(com_term)  
                    - 180*epsilon**2*spin**4*y**11 + 348*epsilon**2*spin**4*y**10  
                    + 24*epsilon**2*spin**3*y**12*numpy.sqrt(com_term)  
                    - 48*epsilon**2*spin**3*y**11*numpy.sqrt(com_term)  
                    - 12*epsilon**2*spin**2*y**14*(com_term) + 18*epsilon**2*spin**2*y**14 
                    + 64*epsilon**2*spin**2*y**13*(com_term) - 210*epsilon**2*spin**2*y**13  
                    - 80*epsilon**2*spin**2*y**12*(com_term) + 648*epsilon**2*spin**2*y**12  
                    - 600*epsilon**2*spin**2*y**11 + 24*epsilon**2*spin*y**14*numpy.sqrt(com_term)  
                    - 96*epsilon**2*spin*y**13*numpy.sqrt(com_term)  
                    + 96*epsilon**2*spin*y**12*numpy.sqrt(com_term)  
                    + 9*epsilon**2*y**16 - 84*epsilon**2*y**15 + 292*epsilon**2*y**14  
                    - 448*epsilon**2*y**13 + 256*epsilon**2*y**12  
                    - 12*epsilon*spin**5*y**13*numpy.sqrt(com_term)  
                    - 6*epsilon*spin**4*y**15* (com_term)  
                    + 24*epsilon*spin**4*y**14*(com_term) - 12*epsilon*spin**4*y**14 
                    + 40*epsilon*spin**4*y**13 - 24*epsilon*spin**3*y**15*numpy.sqrt(com_term) 
                    + 48*epsilon*spin**3*y**14*numpy.sqrt(com_term)  
                    - 12*epsilon*spin**2*y**17*(com_term)  
                    + 68*epsilon*spin**2*y**16*(com_term) - 24*epsilon*spin**2*y**16 
                    - 88*epsilon*spin**2*y**15*(com_term) + 120*epsilon*spin**2*y**15  
                    - 144*epsilon*spin**2*y**14 - 12*epsilon*spin*y**17*numpy.sqrt(com_term) 
                    + 48*epsilon*spin*y**16*numpy.sqrt(com_term) 
                    - 48*epsilon*spin*y**15*numpy.sqrt(com_term)  
                    - 6*epsilon*y**19*   (com_term)  
                    + 44*epsilon*y**18*  (com_term) - 12*epsilon*y**18  
                    - 104*epsilon*y**17* (com_term) + 80*epsilon*y**17 
                    +  80*epsilon*y**16* (com_term) - 176*epsilon*y**16 + 128*epsilon*y**15 
                    + 2*spin**4*y**17* (com_term) + 4*spin**4*y**16  
                    + 4*spin**2*y**19* (com_term)  
                    - 8*spin**2*y**18* (com_term) + 8*spin**2*y**18- 16*spin**2*y**17  
                    + 2*y**21*(com_term)  
                    - 8*y**20*(com_term) + 4*y**20  
                    + 8*y**19*(com_term) - 16*y**19 + 16*y**18) 
        denominator = (y**6*(4*epsilon**4*spin**2 + 16*epsilon**3*spin**2*y**3  
                            + 4*epsilon**3*spin*y**5*numpy.sqrt(com_term)  
                            - 8*epsilon**3*spin*y**4*numpy.sqrt(com_term)+ 24*epsilon**2*spin**2*y**6  
                            + 12*epsilon**2*spin*y**8*numpy.sqrt(com_term)  
                            - 24*epsilon**2*spin*y**7*numpy.sqrt(com_term)  
                            + epsilon**2*y**10*(com_term) 
                            - 4*epsilon**2*y**9*(com_term) 
                            + 4*epsilon**2*y**8*(com_term) + 16*epsilon*spin**2*y**9 
                            + 12*epsilon*spin*y**11*numpy.sqrt(com_term) 
                            - 24*epsilon*spin*y**10*numpy.sqrt(com_term)  
                            + 2*epsilon*y**13*(com_term) 
                            - 8*epsilon*y**12*(com_term)  
                            + 8*epsilon*y**11*(com_term) + 4*spin**2*y**12  
                            + 4*spin*y**14*numpy.sqrt(com_term)  
                            - 8*spin*y**13*numpy.sqrt(com_term)  
                            + y**16*(com_term)  
                            - 4*y**15*(com_term)  
                            + 4*y**14*(com_term))) 

    #         print(com_term,numerator,denominator,np.sqrt(numerator/denominator)) 
        if (numerator/denominator) >= 0: 

            gam = freqM_dim_less(spin,epsilon)*numpy.sqrt(numerator/denominator) 
            
            return -1*(gam/4)
        return -10
    return -10

def frequency_in_hertz(mass, spin, epsilon, l, m, n): 
    
    return m*(freqM_dim_less(spin,epsilon)+ real_beta(spin,l))/(4*numpy.pi*mass*lal.MTSUN_SI) 

def damping_in_seconds(mass, spin, epsilon, l, m, n): 
    return -1*(mass*lal.MTSUN_SI)/(gamma0(spin,epsilon)-im_beta(spin,l))

def get_JP_lm_f0tau(mass,spin,epsilon,l,m,n=0, which='both'):
    mass, spin, l, m, n, input_is_array = ensurearray(
        mass, spin, l, m, n)
    # we'll ravel the arrays so we can evaluate each parameter combination
    # one at a a time
    getf0 = which == 'both' or which == 'f0'
    gettau = which == 'both' or which == 'tau'
    out = []
    if getf0:
        f0s = frequency_in_hertz(mass,spin,epsilon, l, m, n) 
        out.append(formatreturn(f0s, input_is_array))
    if gettau:
        taus = damping_in_seconds(mass,spin,epsilon, l, m, n) 
        out.append(formatreturn(taus, input_is_array))
    if not (getf0 and gettau):
        out = out[0]
    return out
def get_JP_lm_f0tau_allmodes(mass,spin,epsilon,modes):
    f0, tau = {}, {}
    for lmn in modes:
        key = '{}{}{}'
        l, m, nmodes = int(lmn[0]), int(lmn[1]), int(lmn[2])
        for n in range(nmodes):
            tmp_f0, tmp_tau = get_JP_lm_f0tau(mass, spin,epsilon, l, m, n)
            f0[key.format(l, abs(m), n)] = tmp_f0
            tau[key.format(l, abs(m), n)] = tmp_tau
    return f0, tau

def real_beta(spin,l):
    if l == 2:
        a1,a2,a3,a4,a5,a6,err = 0.1282,0.4178,0.6711,0.5037,1.8331,0.7596,0.023
    elif l == 3:
        a1,a2,a3,a4,a5,a6,err = 0.1801, 0.5007,0.7064,0.5704,1.4690,0.7302, 0.005
    elif l == 4:
        a1,a2,a3,a4,a5,a6,err = 0.1974, 0.4982, 0.6808, 0.5958,1.4380, 0.7102,0.011
    elif l == 5:
        a1,a2,a3,a4,a5,a6,err = 0.2083,0.4762, 0.6524, 0.6167, 1.4615, 0.6937,0.016
    elif l == 6:
        a1,a2,a3,a4,a5,a6,err = 0.2167, 0.4458, 0.6235, 0.6373, 1.5103, 0.6791,0.021
    elif l == 7:
        a1,a2,a3,a4,a5,a6,err =0.2234, 0.4116, 0.5933, 0.6576, 1.5762, 0.6638,0.025
        
    return a1 + a2*numpy.exp(-a3*(1-(spin))**a4) - (1/(a5 + (1-(spin))**a6)) + (err*1e-2)


def im_beta(spin,l):
    if l == 2:
        a1,a2,a3,a4,a5,a6,err = 0.1381,0.3131,0.5531,0.8492,2.2159,0.8544,0.004
    elif l == 3:
        a1,a2,a3,a4,a5,a6,err = 0.1590,0.3706,0.6643,0.6460,1.8889,0.6676,0.008
    elif l == 4:
        a1,a2,a3,a4,a5,a6,err = 0.1575,0.3478,0.6577,0.5840,1.9799,0.6032,0.009
    elif l == 5:
        a1,a2,a3,a4,a5,a6,err = 0.1225,0.1993,0.4855,0.6313,3.1018,0.6150,1.335
    elif l == 6:
        a1,a2,a3,a4,a5,a6,err = 0.1280,0.1947,0.5081,0.6556,3.0960,0.6434,0.665
    elif l == 7:
        a1,a2,a3,a4,a5,a6,err = -15.333,15.482,0.0011,0.3347,6.6258,0.2974,0.874

    return a1 + a2*numpy.exp(-a3*(1-(spin))**a4) - (1/(a5 + (1-(spin))**a6)) + (err*1e-2)


####################### JP Ringdown Functions end   #####################




# def get_JP_lm_f0tau(mass,spin,epsilon,l,m,n=0, which='both'):
#     #returns frequency and damping time for JP geometry
#     #Only valid for n=0 modes
#     #f1 = 1.5251
#     #f2 = -1.1568
#     #f3 = 0.1292

#     #q1 = 0.7000
#     #q2 = 1.4187
#     #q3 = -0.4990
#     # convert to arrays
#     mass, spin, epsilon, l, m, n, input_is_array = ensurearray(
#         mass, spin, epsilon, l, m, n)
#     # we'll ravel the arrays so we can evaluate each parameter combination
#     # one at a a time
#     ########################################## complete expressions #########################################
    
#     getf0 = which == 'both' or which == 'f0'
#     gettau = which == 'both' or which == 'tau'
#     out = []
#     mass = mass*lal.MTSUN_SI
#     if spin>0:
#         a = spin*mass
#         r_K = 2*mass*(1 + numpy.cos((2/3)*numpy.arccos(-a/mass)))
#         w_K = (mass**(0.5))/((r_K**(1.5) + a*mass**(0.5)))


#         #w_K = (f1+f2*((1.-x)**f3))/(2*np.pi*m)
#         b_ph = 1/(w_K)
#         gam_ph = 2*(3*mass)**0.5 * ((r_K**2 -2*mass*r_K + (a)**2)*w_K) / (r_K**1.5*(r_K - mass))
        
#         C1 = a + b_ph
#         C2 = a - b_ph

#         C0 = 27*mass**2*C2*(4*a + b_ph) + 2*C1**4

#         d_b1 = (54*mass**2*C2*C1**4 + C1**7) / (54* C2**2 * C0)

#         #d_b2 = (C1**7/(1944*C2**5*C0**3))*(78732*mass**6*C2**3*(29*a**2 +4*a*b_ph - b_ph**2) + 729*mass**4*C2**2*C1**3*(204*a**2 + 88*a*b_ph + b_ph**2) + 27*mass**2*C2*C1**6*(117*a**2 + 96*a*b_ph + 13*b_ph**2) + 2*C1**9*(11*a**2 + 14*a*b_ph + 4*b_ph**2))
        
#         d_gam1 = (gam_ph**3)*((mass**4*C1**2*(27*mass**2*C2**2 + a*(a - 2*b_ph))/(2*C1**5 * C0*(3*mass**2*C2*(5*a - b_ph) + a**2*C1**2)**3)))*((C1**(10))*a**3*(4*a**2 - a*b_ph - 6*b_ph**2) + 729*mass**6*C2**3*C1**2*(364*a**4 - 227*a**3*b_ph - 201*a**2*b_ph**2 - 29*a*b_ph**3 + 13*b_ph**4) + 27*mass**4*C2**2*(2*a+b_ph)*(319*a**4 - 174*a**3*b_ph - 216*a**2*b_ph**2 - 38*a*b_ph**3 + 9*b_ph**4)*C1**4 + a*mass**2*C2*C1**7*(454*a**4 - 133*a**3*b_ph - 366*a**2*b_ph**2 - 182*a*b_ph**3 + 2*b_ph**4) + 78732*mass**8*C2**5*(5*a - b_ph)*(4*a + b_ph)) 

        
#         #f0s= l*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon - ((b_ph*d_b2 - d_b1**2)/(b_ph**3))*2*epsilon**2)/(2*numpy.pi)
#         f0s= l*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon)/(2*numpy.pi)
#         taus = (2)/(gam_ph + epsilon*(d_gam1))
#         if taus>0 and f0s>0:
#             if getf0:
#                 out.append(formatreturn(f0s, input_is_array))
#             if gettau:
#                 out.append(formatreturn(taus, input_is_array))
#             if not (getf0 and gettau):
#                 out = out[0]
#             return out
#         else:
#             f0s= numpy.zeros((1,)) #m*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon - ((b_ph*d_b2 - d_b1**2)/(b_ph**3))*2*epsilon**2)/(2*np.pi)
#             taus = abs(taus)#numpy.full((1,), numpy.inf) #(2)/(gam_ph + epsilon*(d_gam1))
#             if getf0:
#                 out.append(formatreturn(f0s, input_is_array))
#             if gettau:
#                 out.append(formatreturn(taus, input_is_array))
#             if not (getf0 and gettau):
#                 out = out[0]
#             return out
#     else:
#         a = spin*mass
#         r_K = 2*mass*(1 + numpy.cos((2/3)*numpy.arccos(a/mass)))
#         w_K = (mass**(0.5))/((r_K**(1.5) - a*mass**(0.5)))

#         b_ph = 1/(w_K)
#         gam_ph = 2*(3*mass)**0.5 * ((r_K**2 -2*mass*r_K + (a)**2)*(w_K)) / (r_K**1.5*(r_K - mass))
        
#         C1 = a + b_ph
#         C2 = a - b_ph
#         C0 = 27*mass**2*C2*(4*a + b_ph) + 2*C1**4

#         d_b1 = (54*mass**2*C2*C1**4 + C1**7) / (54* C2**2 * C0)

#         #d_b2 = (C1**7/(1944*C2**5*C0**3))*(78732*mass**6*C2**3*(29*a**2 +4*a*b_ph - b_ph**2) + 729*mass**4*C2**2*C1**3*(204*a**2 + 88*a*b_ph + b_ph**2) + 27*mass**2*C2*C1**6*(117*a**2 + 96*a*b_ph + 13*b_ph**2) + 2*C1**9*(11*a**2 + 14*a*b_ph + 4*b_ph**2))
        
#         d_gam1 = (gam_ph**3)*((mass**4*C1**2*(27*mass**2*C2**2 + a*(a - 2*b_ph))/(2*C1**5 * C0*(3*mass**2*C2*(5*a - b_ph) + a**2*C1**2)**3)))*((C1**(10))*a**3*(4*a**2 - a*b_ph - 6*b_ph**2) + 729*mass**6*C2**3*C1**2*(364*a**4 - 227*a**3*b_ph - 201*a**2*b_ph**2 - 29*a*b_ph**3 + 13*b_ph**4) + 27*mass**4*C2**2*(2*a+b_ph)*(319*a**4 - 174*a**3*b_ph - 216*a**2*b_ph**2 - 38*a*b_ph**3 + 9*b_ph**4)*C1**4 + a*mass**2*C2*C1**7*(454*a**4 - 133*a**3*b_ph - 366*a**2*b_ph**2 - 182*a*b_ph**3 + 2*b_ph**4) + 78732*mass**8*C2**5*(5*a - b_ph)*(4*a + b_ph)) 
        
    
#         #f0s = -l*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon - ((b_ph*d_b2 - d_b1**2)/(b_ph**3))*2*epsilon**2)/(2*numpy.pi)
#         f0s= l*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon)/(2*numpy.pi)
#         taus = (2)/(gam_ph + epsilon*(d_gam1))
#         if taus>0 and f0s>0:
#             if getf0:
#                 out.append(formatreturn(f0s, input_is_array))
#             if gettau:
#                 out.append(formatreturn(taus, input_is_array))
#             if not (getf0 and gettau):
#                 out = out[0]
#             return out
#         else:
#             f0s= numpy.zeros((1,)) #m*((1/b_ph) - ((2*d_b1)/(b_ph**2))*epsilon - ((b_ph*d_b2 - d_b1**2)/(b_ph**3))*2*epsilon**2)/(2*np. pi)
#             taus = abs(taus) #numpy.full((1,), numpy.inf) #(2)/(gam_ph + epsilon*(d_gam1))
#             if getf0:
#                 out.append(formatreturn(f0s, input_is_array))
#             if gettau:
#                 out.append(formatreturn(taus, input_is_array))
#             if not (getf0 and gettau):
#                 out = out[0]
#             return out
#     ########################################## complete expressions ###################################################
#     #getf0 = which == 'both' or which == 'f0'
#     #gettau = which == 'both' or which == 'tau'
#     #out = []
#     #if getf0:
#     #    if m>0:
#     #        f0s = pykerr.qnmfreq(mass, spin, l, m, n) + ((l*epsilon*((1/(81.*numpy.sqrt(3))+((10.*spin)/(729.))+((47.*spin**2)/(1458.*numpy.sqrt(3))))))/(2.*numpy.pi*mass*lal.MTSUN_SI))
#      #       out.append(formatreturn(f0s, input_is_array))
#      #   else:
#      #       f0s = pykerr.qnmfreq(mass, spin, l, m, n) - ((l*epsilon*((1/(81.*numpy.sqrt(3))+((10.*spin)/(729.))+((47.*  spin**2)/(1458.*numpy.sqrt(3))))))/(2.*numpy.pi*mass*lal.MTSUN_SI))
#       #      out.append(formatreturn(f0s, input_is_array))
#     #if gettau:
#     #    taus = 1 / ((1 /pykerr.qnmtau(mass, spin, l, m, n)) + ((epsilon*((spin)/(486.) + spin**2*(16.)/(2187.*numpy.    sqrt(3))))/(mass*lal.MTSUN_SI)))
#      #   out.append(formatreturn(taus, input_is_array))
#         #if epsilon == 0. or spin == 0:
#         #    taus=pykerr.qnmtau(mass, spin, l, m, n)
#         #    out.append(formatreturn(taus, input_is_array))
#         #else:
#             #taus = pykerr.qnmtau(mass, spin, l, m, n) - 2.*numpy.pi*mass*lal.MTSUN_SI*(1/epsilon)*(2*n + 1)*(1/((spin)/(486.) + spin**2*(16.)/(2187.*numpy.sqrt(3))))
#         #taus = 1 / ((1 /pykerr.qnmtau(mass, spin, l, m, n)) + ((epsilon*((spin)/(486.) + spin**2*(16.)/(2187.*numpy. sqrt(3))))/(mass*lal.MTSUN_SI)))
#     #if not (getf0 and gettau):
#     #    out = out[0]
#     #return out
#     #f0=(((f1+f2*((1.-spin)**f3)))/(2.*numpy.pi*mass) + (l*epsilon*((1/(81.*np.sqrt(3))+((10.*spin)/(729.))+((47.*spin**2)/(1458.*numpy.sqrt(3))))))/(2.*numpy.pi*mass))/lal.MTSUN_SI
#     #tau = (((1/(2.*numpy.pi*mass))*(((f1 + f2*((1.-spin)**f3))*numpy.pi)/(q1 + q2*((1-spin)**q3)) - epsilon*((spin)/(486.) + (16.*spin**2)/(2187.*numpy.sqrt(3))))/lal.MTSUN_SI))**(-1)
#     #return f0, tau

# def get_JP_lm_f0tau_allmodes(mass,spin,epsilon,modes):
#     f0, tau = {}, {}
#     for lmn in modes:
#         key = '{}{}{}'
#         l, m, nmodes = int(lmn[0]), int(lmn[1]), int(lmn[2])
#         for n in range(nmodes):
#             tmp_f0, tmp_tau = get_JP_lm_f0tau(mass, spin,epsilon, l, m, n)
#             f0[key.format(l, abs(m), n)] = tmp_f0
#             tau[key.format(l, abs(m), n)] = tmp_tau
#     return f0, tau

def get_lm_f0tau_allmodes(mass, spin, modes):
    """Returns a dictionary of all of the frequencies and damping times for the
    requested modes.

    Parameters
    ----------
    mass : float or array
        Mass of the black hole (in solar masses).
    spin : float or array
        Dimensionless spin of the final black hole.
    modes : list of str
        The modes to get. Each string in the list should be formatted
        'lmN', where l (m) is the l (m) index of the harmonic and N is the
        number of overtones to generate (note, N is not the index of the
        overtone).

    Returns
    -------
    f0 : dict
        Dictionary mapping the modes to the frequencies. The dictionary keys
        are 'lmn' string, where l (m) is the l (m) index of the harmonic and
        n is the index of the overtone. For example, '220' is the l = m = 2
        mode and the 0th overtone.
    tau : dict
        Dictionary mapping the modes to the damping times. The keys are the
        same as ``f0``.
    """
    f0, tau = {}, {}
    for lmn in modes:
        key = '{}{}{}'
        l, m, nmodes = int(lmn[0]), int(lmn[1]), int(lmn[2])
        for n in range(nmodes):
            tmp_f0, tmp_tau = get_lm_f0tau(mass, spin, l, m, n)
            f0[key.format(l, abs(m), n)] = tmp_f0
            tau[key.format(l, abs(m), n)] = tmp_tau
    return f0, tau


def freq_from_final_mass_spin(final_mass, final_spin, l=2, m=2, n=0):
    """Returns QNM frequency for the given mass and spin and mode.

    Parameters
    ----------
    final_mass : float or array
        Mass of the black hole (in solar masses).
    final_spin : float or array
        Dimensionless spin of the final black hole.
    l : int or array, optional
        l-index of the harmonic. Default is 2.
    m : int or array, optional
        m-index of the harmonic. Default is 2.
    n : int or array
        Overtone(s) to generate, where n=0 is the fundamental mode.
        Default is 0.

    Returns
    -------
    float or array
        The frequency of the QNM(s), in Hz.
    """
    return get_lm_f0tau(final_mass, final_spin, l, m, n=n, which='f0')


def tau_from_final_mass_spin(final_mass, final_spin, l=2, m=2, n=0):
    """Returns QNM damping time for the given mass and spin and mode.

    Parameters
    ----------
    final_mass : float or array
        Mass of the black hole (in solar masses).
    final_spin : float or array
        Dimensionless spin of the final black hole.
    l : int or array, optional
        l-index of the harmonic. Default is 2.
    m : int or array, optional
        m-index of the harmonic. Default is 2.
    n : int or array
        Overtone(s) to generate, where n=0 is the fundamental mode.
        Default is 0.

    Returns
    -------
    float or array
        The damping time of the QNM(s), in seconds.
    """
    return get_lm_f0tau(final_mass, final_spin, l, m, n=n, which='tau')


# The following are from Table VIII, IX, X of Berti et al.,
# PRD 73 064030, arXiv:gr-qc/0512160 (2006).
# Keys are l,m (only n=0 supported). Constants are for converting from
# frequency and damping time to mass and spin.
_berti_spin_constants = {
    (2, 2): (0.7, 1.4187, -0.4990),
    (2, 1): (-0.3, 2.3561, -0.2277),
    (3, 3): (0.9, 2.343, -0.4810),
    (4, 4): (1.1929, 3.1191, -0.4825),
    }

_berti_mass_constants = {
    (2, 2): (1.5251, -1.1568, 0.1292),
    (2, 1): (0.6, -0.2339, 0.4175),
    (3, 3): (1.8956, -1.3043, 0.1818),
    (4, 4): (2.3, -1.5056, 0.2244),
    }


def final_spin_from_f0_tau(f0, tau, l=2, m=2):
    """Returns the final spin based on the given frequency and damping time.

    .. note::
        Currently, only (l,m) = (2,2), (3,3), (4,4), (2,1) are supported.
        Any other indices will raise a ``KeyError``.

    Parameters
    ----------
    f0 : float or array
        Frequency of the QNM (in Hz).
    tau : float or array
        Damping time of the QNM (in seconds).
    l : int, optional
        l-index of the harmonic. Default is 2.
    m : int, optional
        m-index of the harmonic. Default is 2.

    Returns
    -------
    float or array
        The spin of the final black hole. If the combination of frequency
        and damping times give an unphysical result, ``numpy.nan`` will be
        returned.
    """
    f0, tau, input_is_array = ensurearray(f0, tau)
    # from Berti et al. 2006
    a, b, c = _berti_spin_constants[l,m]
    origshape = f0.shape
    # flatten inputs for storing results
    f0 = f0.ravel()
    tau = tau.ravel()
    spins = numpy.zeros(f0.size)
    for ii in range(spins.size):
        Q = f0[ii] * tau[ii] * numpy.pi
        try:
            s = 1. - ((Q-a)/b)**(1./c)
        except ValueError:
            s = numpy.nan
        spins[ii] = s
    spins = spins.reshape(origshape)
    return formatreturn(spins, input_is_array)


def final_mass_from_f0_tau(f0, tau, l=2, m=2):
    """Returns the final mass (in solar masses) based on the given frequency
    and damping time.

    .. note::
        Currently, only (l,m) = (2,2), (3,3), (4,4), (2,1) are supported.
        Any other indices will raise a ``KeyError``.

    Parameters
    ----------
    f0 : float or array
        Frequency of the QNM (in Hz).
    tau : float or array
        Damping time of the QNM (in seconds).
    l : int, optional
        l-index of the harmonic. Default is 2.
    m : int, optional
        m-index of the harmonic. Default is 2.

    Returns
    -------
    float or array
        The mass of the final black hole. If the combination of frequency
        and damping times give an unphysical result, ``numpy.nan`` will be
        returned.
    """
    # from Berti et al. 2006
    spin = final_spin_from_f0_tau(f0, tau, l=l, m=m)
    a, b, c = _berti_mass_constants[l,m]
    return (a + b*(1-spin)**c)/(2*numpy.pi*f0*lal.MTSUN_SI)

def freqlmn_from_other_lmn(f0, tau, current_l, current_m, new_l, new_m):
    """Returns the QNM frequency (in Hz) of a chosen new (l,m) mode from the
    given current (l,m) mode.

    Parameters
    ----------
    f0 : float or array
        Frequency of the current QNM (in Hz).
    tau : float or array
        Damping time of the current QNM (in seconds).
    current_l : int, optional
        l-index of the current QNM.
    current_m : int, optional
        m-index of the current QNM.
    new_l : int, optional
        l-index of the new QNM to convert to.
    new_m : int, optional
        m-index of the new QNM to convert to.

    Returns
    -------
    float or array
        The frequency of the new (l, m) QNM mode. If the combination of
        frequency and damping time provided for the current (l, m) QNM mode
        correspond to an unphysical Kerr black hole mass and/or spin,
        ``numpy.nan`` will be returned.
    """
    mass = final_mass_from_f0_tau(f0, tau, l=current_l, m=current_m)
    spin = final_spin_from_f0_tau(f0, tau, l=current_l, m=current_m)
    mass, spin, input_is_array = ensurearray(mass, spin)

    mass[mass < 0] = numpy.nan
    spin[numpy.abs(spin) > 0.9996] = numpy.nan

    new_f0 = freq_from_final_mass_spin(mass, spin, l=new_l, m=new_m)
    return formatreturn(new_f0, input_is_array)


def taulmn_from_other_lmn(f0, tau, current_l, current_m, new_l, new_m):
    """Returns the QNM damping time (in seconds) of a chosen new (l,m) mode
    from the given current (l,m) mode.

    Parameters
    ----------
    f0 : float or array
        Frequency of the current QNM (in Hz).
    tau : float or array
        Damping time of the current QNM (in seconds).
    current_l : int, optional
        l-index of the current QNM.
    current_m : int, optional
        m-index of the current QNM.
    new_l : int, optional
        l-index of the new QNM to convert to.
    new_m : int, optional
        m-index of the new QNM to convert to.

    Returns
    -------
    float or array
        The daming time of the new (l, m) QNM mode. If the combination of
        frequency and damping time provided for the current (l, m) QNM mode
        correspond to an unphysical Kerr black hole mass and/or spin,
        ``numpy.nan`` will be returned.
    """
    mass = final_mass_from_f0_tau(f0, tau, l=current_l, m=current_m)
    spin = final_spin_from_f0_tau(f0, tau, l=current_l, m=current_m)
    mass, spin, input_is_array = ensurearray(mass, spin)

    mass[mass < 0] = numpy.nan
    spin[numpy.abs(spin) > 0.9996] = numpy.nan

    new_tau = tau_from_final_mass_spin(mass, spin, l=new_l, m=new_m)
    return formatreturn(new_tau, input_is_array)

def get_final_from_initial(mass1, mass2, spin1x=0., spin1y=0., spin1z=0.,
                           spin2x=0., spin2y=0., spin2z=0.,
                           approximant='SEOBNRv4PHM', f_ref=-1):
    """Estimates the final mass and spin from the given initial parameters.

    This uses the fits used by either the NRSur7dq4 or EOBNR models for
    converting from initial parameters to final, depending on the
    ``approximant`` argument.

    Parameters
    ----------
    mass1 : float
        The mass of one of the components, in solar masses.
    mass2 : float
        The mass of the other component, in solar masses.
    spin1x : float, optional
        The dimensionless x-component of the spin of mass1. Default is 0.
    spin1y : float, optional
        The dimensionless y-component of the spin of mass1. Default is 0.
    spin1z : float, optional
        The dimensionless z-component of the spin of mass1. Default is 0.
    spin2x : float, optional
        The dimensionless x-component of the spin of mass2. Default is 0.
    spin2y : float, optional
        The dimensionless y-component of the spin of mass2. Default is 0.
    spin2z : float, optional
        The dimensionless z-component of the spin of mass2. Default is 0.
    approximant : str, optional
        The waveform approximant to use for the fit function. If "NRSur7dq4",
        the NRSur7dq4Remnant fit in lalsimulation will be used. If "SEOBNRv4",
        the ``XLALSimIMREOBFinalMassSpin`` function in lalsimulation will be
        used. Otherwise, ``XLALSimIMREOBFinalMassSpinPrec`` from lalsimulation
        will be used, with the approximant name passed as the approximant
        in that function ("SEOBNRv4PHM" will work with this function).
        Default is "SEOBNRv4PHM".
    f_ref : float, optional
        The reference frequency for the spins. Only used by the NRSur7dq4
        fit. Default (-1) will use the default reference frequency for the
        approximant.

    Returns
    -------
    final_mass : float
        The final mass, in solar masses.
    final_spin : float
        The dimensionless final spin.
    """
    args = (mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z)
    args = ensurearray(*args)
    input_is_array = args[-1]
    origshape = args[0].shape
    # flatten inputs for storing results
    args = [a.ravel() for a in args[:-1]]
    mass1, mass2, spin1x, spin1y, spin1z, spin2x, spin2y, spin2z = args
    final_mass = numpy.full(mass1.shape, numpy.nan)
    final_spin = numpy.full(mass1.shape, numpy.nan)
    for ii in range(final_mass.size):
        m1 = float(mass1[ii])
        m2 = float(mass2[ii])
        spin1 = list(map(float, [spin1x[ii], spin1y[ii], spin1z[ii]]))
        spin2 = list(map(float, [spin2x[ii], spin2y[ii], spin2z[ii]]))
        if approximant == 'NRSur7dq4':
            from lalsimulation import nrfits
            try:
                res = nrfits.eval_nrfit(m1*lal.MSUN_SI,
                                        m2*lal.MSUN_SI,
                                        spin1, spin2, 'NRSur7dq4Remnant',
                                        ['FinalMass', 'FinalSpin'],
                                        f_ref=f_ref)
            except RuntimeError:
                continue
            final_mass[ii] = res['FinalMass'][0] / lal.MSUN_SI
            sf = res['FinalSpin']
            final_spin[ii] = (sf**2).sum()**0.5
            if sf[-1] < 0:
                final_spin[ii] *= -1
        elif approximant == 'SEOBNRv4':
            _, fm, fs = lalsim.SimIMREOBFinalMassSpin(
                m1, m2, spin1, spin2, getattr(lalsim, approximant))
            final_mass[ii] = fm * (m1 + m2)
            final_spin[ii] = fs
        else:
            _, fm, fs = lalsim.SimIMREOBFinalMassSpinPrec(
                m1, m2, spin1, spin2, getattr(lalsim, approximant))
            final_mass[ii] = fm * (m1 + m2)
            final_spin[ii] = fs
    final_mass = final_mass.reshape(origshape)
    final_spin = final_spin.reshape(origshape)
    return (formatreturn(final_mass, input_is_array),
            formatreturn(final_spin, input_is_array))


def final_mass_from_initial(mass1, mass2, spin1x=0., spin1y=0., spin1z=0.,
                            spin2x=0., spin2y=0., spin2z=0.,
                            approximant='SEOBNRv4PHM', f_ref=-1):
    """Estimates the final mass from the given initial parameters.

    This uses the fits used by either the NRSur7dq4 or EOBNR models for
    converting from initial parameters to final, depending on the
    ``approximant`` argument.

    Parameters
    ----------
    mass1 : float
        The mass of one of the components, in solar masses.
    mass2 : float
        The mass of the other component, in solar masses.
    spin1x : float, optional
        The dimensionless x-component of the spin of mass1. Default is 0.
    spin1y : float, optional
        The dimensionless y-component of the spin of mass1. Default is 0.
    spin1z : float, optional
        The dimensionless z-component of the spin of mass1. Default is 0.
    spin2x : float, optional
        The dimensionless x-component of the spin of mass2. Default is 0.
    spin2y : float, optional
        The dimensionless y-component of the spin of mass2. Default is 0.
    spin2z : float, optional
        The dimensionless z-component of the spin of mass2. Default is 0.
    approximant : str, optional
        The waveform approximant to use for the fit function. If "NRSur7dq4",
        the NRSur7dq4Remnant fit in lalsimulation will be used. If "SEOBNRv4",
        the ``XLALSimIMREOBFinalMassSpin`` function in lalsimulation will be
        used. Otherwise, ``XLALSimIMREOBFinalMassSpinPrec`` from lalsimulation
        will be used, with the approximant name passed as the approximant
        in that function ("SEOBNRv4PHM" will work with this function).
        Default is "SEOBNRv4PHM".
    f_ref : float, optional
        The reference frequency for the spins. Only used by the NRSur7dq4
        fit. Default (-1) will use the default reference frequency for the
        approximant.

    Returns
    -------
    float
        The final mass, in solar masses.
    """
    return get_final_from_initial(mass1, mass2, spin1x, spin1y, spin1z,
                                  spin2x, spin2y, spin2z, approximant,
                                  f_ref=f_ref)[0]


def final_spin_from_initial(mass1, mass2, spin1x=0., spin1y=0., spin1z=0.,
                            spin2x=0., spin2y=0., spin2z=0.,
                            approximant='SEOBNRv4PHM', f_ref=-1):
    """Estimates the final spin from the given initial parameters.

    This uses the fits used by either the NRSur7dq4 or EOBNR models for
    converting from initial parameters to final, depending on the
    ``approximant`` argument.

    Parameters
    ----------
    mass1 : float
        The mass of one of the components, in solar masses.
    mass2 : float
        The mass of the other component, in solar masses.
    spin1x : float, optional
        The dimensionless x-component of the spin of mass1. Default is 0.
    spin1y : float, optional
        The dimensionless y-component of the spin of mass1. Default is 0.
    spin1z : float, optional
        The dimensionless z-component of the spin of mass1. Default is 0.
    spin2x : float, optional
        The dimensionless x-component of the spin of mass2. Default is 0.
    spin2y : float, optional
        The dimensionless y-component of the spin of mass2. Default is 0.
    spin2z : float, optional
        The dimensionless z-component of the spin of mass2. Default is 0.
    approximant : str, optional
        The waveform approximant to use for the fit function. If "NRSur7dq4",
        the NRSur7dq4Remnant fit in lalsimulation will be used. If "SEOBNRv4",
        the ``XLALSimIMREOBFinalMassSpin`` function in lalsimulation will be
        used. Otherwise, ``XLALSimIMREOBFinalMassSpinPrec`` from lalsimulation
        will be used, with the approximant name passed as the approximant
        in that function ("SEOBNRv4PHM" will work with this function).
        Default is "SEOBNRv4PHM".
    f_ref : float, optional
        The reference frequency for the spins. Only used by the NRSur7dq4
        fit. Default (-1) will use the default reference frequency for the
        approximant.

    Returns
    -------
    float
        The dimensionless final spin.
    """
    return get_final_from_initial(mass1, mass2, spin1x, spin1y, spin1z,
                                  spin2x, spin2y, spin2z, approximant,
                                  f_ref=f_ref)[1]


#
# =============================================================================
#
#                         post-Newtonian functions
#
# =============================================================================
#

def velocity_to_frequency(v, M):
    """ Calculate the gravitational-wave frequency from the
    total mass and invariant velocity.

    Parameters
    ----------
    v : float
        Invariant velocity
    M : float
        Binary total mass

    Returns
    -------
    f : float
        Gravitational-wave frequency
    """
    return v**(3.0) / (M * lal.MTSUN_SI * lal.PI)

def frequency_to_velocity(f, M):
    """ Calculate the invariant velocity from the total
    mass and gravitational-wave frequency.

    Parameters
    ----------
    f: float
        Gravitational-wave frequency
    M: float
        Binary total mass

    Returns
    -------
    v : float or numpy.array
        Invariant velocity
    """
    return (lal.PI * M * lal.MTSUN_SI * f)**(1.0/3.0)


def f_schwarzchild_isco(M):
    """
    Innermost stable circular orbit (ISCO) for a test particle
    orbiting a Schwarzschild black hole

    Parameters
    ----------
    M : float or numpy.array
        Total mass in solar mass units

    Returns
    -------
    f : float or numpy.array
        Frequency in Hz
    """
    return velocity_to_frequency((1.0/6.0)**(0.5), M)


#
# ============================================================================
#
#                          p-g mode non-linear tide functions
#
# ============================================================================
#

def nltides_coefs(amplitude, n, m1, m2):
    """Calculate the coefficents needed to compute the
    shift in t(f) and phi(f) due to non-linear tides.

    Parameters
    ----------
    amplitude: float
        Amplitude of effect
    n: float
        Growth dependence of effect
    m1: float
        Mass of component 1
    m2: float
        Mass of component 2

    Returns
    -------
    f_ref : float
        Reference frequency used to define A and n
    t_of_f_factor: float
        The constant factor needed to compute t(f)
    phi_of_f_factor: float
        The constant factor needed to compute phi(f)
    """

    # Use 100.0 Hz as a reference frequency
    f_ref = 100.0

    # Calculate chirp mass
    mc = mchirp_from_mass1_mass2(m1, m2)
    mc *= lal.lal.MSUN_SI

    # Calculate constants in phasing
    a = (96./5.) * \
        (lal.lal.G_SI * lal.lal.PI * mc * f_ref / lal.lal.C_SI**3.)**(5./3.)
    b = 6. * amplitude
    t_of_f_factor = -1./(lal.lal.PI*f_ref) * b/(a*a * (n-4.))
    phi_of_f_factor = -2.*b / (a*a * (n-3.))

    return f_ref, t_of_f_factor, phi_of_f_factor


def nltides_gw_phase_difference(f, f0, amplitude, n, m1, m2):
    """Calculate the gravitational-wave phase shift bwtween
    f and f_coalescence = infinity due to non-linear tides.
    To compute the phase shift between e.g. f_low and f_isco,
    call this function twice and compute the difference.

    Parameters
    ----------
    f: float or numpy.array
        Frequency from which to compute phase
    f0: float or numpy.array
        Frequency that NL effects switch on
    amplitude: float or numpy.array
        Amplitude of effect
    n: float or numpy.array
        Growth dependence of effect
    m1: float or numpy.array
        Mass of component 1
    m2: float or numpy.array
        Mass of component 2

    Returns
    -------
    delta_phi: float or numpy.array
        Phase in radians
    """
    f, f0, amplitude, n, m1, m2, input_is_array = ensurearray(
        f, f0, amplitude, n, m1, m2)

    delta_phi = numpy.zeros(m1.shape)

    f_ref, _, phi_of_f_factor = nltides_coefs(amplitude, n, m1, m2)

    mask = f <= f0
    delta_phi[mask] = - phi_of_f_factor[mask] * (f0[mask]/f_ref)**(n[mask]-3.)

    mask = f > f0
    delta_phi[mask] = - phi_of_f_factor[mask] * (f[mask]/f_ref)**(n[mask]-3.)

    return formatreturn(delta_phi, input_is_array)


def nltides_gw_phase_diff_isco(f_low, f0, amplitude, n, m1, m2):
    """Calculate the gravitational-wave phase shift bwtween
    f_low and f_isco due to non-linear tides.

    Parameters
    ----------
    f_low: float
        Frequency from which to compute phase. If the other
        arguments are passed as numpy arrays then the value
        of f_low is duplicated for all elements in the array
    f0: float or numpy.array
        Frequency that NL effects switch on
    amplitude: float or numpy.array
        Amplitude of effect
    n: float or numpy.array
        Growth dependence of effect
    m1: float or numpy.array
        Mass of component 1
    m2: float or numpy.array
        Mass of component 2

    Returns
    -------
    delta_phi: float or numpy.array
        Phase in radians
    """
    f0, amplitude, n, m1, m2, input_is_array = ensurearray(
        f0, amplitude, n, m1, m2)

    f_low = numpy.zeros(m1.shape) + f_low

    phi_l = nltides_gw_phase_difference(
                f_low, f0, amplitude, n, m1, m2)

    f_isco = f_schwarzchild_isco(m1+m2)

    phi_i = nltides_gw_phase_difference(
                f_isco, f0, amplitude, n, m1, m2)

    return formatreturn(phi_i - phi_l, input_is_array)


__all__ = ['dquadmon_from_lambda', 'lambda_tilde',
           'lambda_from_mass_tov_file', 'primary_mass',
           'secondary_mass', 'mtotal_from_mass1_mass2',
           'q_from_mass1_mass2', 'invq_from_mass1_mass2',
           'eta_from_mass1_mass2', 'mchirp_from_mass1_mass2',
           'mass1_from_mtotal_q', 'mass2_from_mtotal_q',
           'mass1_from_mtotal_eta', 'mass2_from_mtotal_eta',
           'mtotal_from_mchirp_eta', 'mass1_from_mchirp_eta',
           'mass2_from_mchirp_eta', 'mass2_from_mchirp_mass1',
           'mass_from_knownmass_eta', 'mass2_from_mass1_eta',
           'mass1_from_mass2_eta', 'eta_from_q', 'mass1_from_mchirp_q',
           'mass2_from_mchirp_q', 'tau0_from_mtotal_eta',
           'tau3_from_mtotal_eta', 'tau0_from_mass1_mass2',
           'tau3_from_mass1_mass2', 'mtotal_from_tau0_tau3',
           'eta_from_tau0_tau3', 'mass1_from_tau0_tau3',
           'mass2_from_tau0_tau3', 'primary_spin', 'secondary_spin',
           'chi_eff', 'chi_a', 'chi_p', 'phi_a', 'phi_s',
           'primary_xi', 'secondary_xi',
           'xi1_from_spin1x_spin1y', 'xi2_from_mass1_mass2_spin2x_spin2y',
           'chi_perp_from_spinx_spiny', 'chi_perp_from_mass1_mass2_xi2',
           'chi_p_from_xi1_xi2', 'phi_from_spinx_spiny',
           'phi1_from_phi_a_phi_s', 'phi2_from_phi_a_phi_s',
           'spin1z_from_mass1_mass2_chi_eff_chi_a',
           'spin2z_from_mass1_mass2_chi_eff_chi_a',
           'spin1x_from_xi1_phi_a_phi_s', 'spin1y_from_xi1_phi_a_phi_s',
           'spin2x_from_mass1_mass2_xi2_phi_a_phi_s',
           'spin2y_from_mass1_mass2_xi2_phi_a_phi_s',
           'chirp_distance', 'det_tc', 'snr_from_loglr',
           'freq_from_final_mass_spin', 'tau_from_final_mass_spin',
           'final_spin_from_f0_tau', 'final_mass_from_f0_tau',
           'final_mass_from_initial', 'final_spin_from_initial',
           'optimal_dec_from_detector', 'optimal_ra_from_detector',
           'chi_eff_from_spherical', 'chi_p_from_spherical',
           'nltides_gw_phase_diff_isco', 'spin_from_pulsar_freq',
           'freqlmn_from_other_lmn', 'taulmn_from_other_lmn',
           'remnant_mass_from_mass1_mass2_spherical_spin_eos',
           'remnant_mass_from_mass1_mass2_cartesian_spin_eos'
          ]
