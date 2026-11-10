import os

import numpy as np
import pytest
from astropy.table import Table
from numpy import allclose

from stpsf import measure_fwhm, roman

GRISM_FILTERS = roman.GRISM_FILTERS
PRISM_FILTERS = roman.PRISM_FILTERS


def pupil_path(wfi, mask=None):
    """
    dynamically generate current pupil path for a given WFI instance
    """
    mask = wfi._pupil_controller._get_pupil_mask(wfi.filter) if mask is None else mask

    base = wfi._pupil_controller._pupil_basepath
    file = wfi._pupil_controller.pupil_file_formatter(mask, wfi.detector)

    return os.path.join(base, file)


def test_WFI_psf():
    """
    Test that instantiating WFI works and can compute a PSF without
    raising any exceptions
    """
    wfi = roman.WFI()
    wfi.calc_psf(fov_pixels=4)


def test_WFI_filters():
    wfi = roman.WFI()
    filter_list = wfi.filter_list

    for fltr in filter_list:
        wfi.filter = fltr
        wfi.calc_psf(fov_pixels=4, oversample=1, nlambda=3)


def test_aberration_detector_position_setter():
    detector = roman.FieldDependentAberration(4096, 4096)

    with pytest.raises(ValueError) as excinfo:
        detector.field_position = (-1, 1)
    assert 'pixel_x' in str(excinfo.value), 'Failed to raise exception for small out-of-bounds ' 'x pixel position'
    with pytest.raises(ValueError) as excinfo:
        detector.field_position = (4096 + 1, 1)
    assert 'pixel_x' in str(excinfo.value), 'Failed to raise exception for large out-of-bounds ' 'x pixel position'
    with pytest.raises(ValueError) as excinfo:
        detector.field_position = (1, -1)
    assert 'pixel_y' in str(excinfo.value), 'Failed to raise exception for small out-of-bounds ' 'y pixel position'
    with pytest.raises(ValueError) as excinfo:
        detector.field_position = (1, 4096 + 1)
    assert 'pixel_y' in str(excinfo.value), 'Failed to raise exception for large out-of-bounds ' 'y pixel position'

    valid_pos = (1.0, 1.0)
    detector.field_position = valid_pos
    assert detector._field_position == valid_pos, (
        'Setting field position through setter did not ' 'update private `_field_position` value'
    )


def test_WFI_fwhm():
    """
    Test that computed PSFs are physically realistic, at least relatively.
    Loose test...
    """
    wfi = roman.WFI()

    wfi._pupilopd = None
    wfi.options['jitter'] = None

    wfi.filter = 'F062'
    fwhm_f062 = measure_fwhm(wfi.calc_psf(oversample=6))

    wfi.filter = 'F184'
    fwhm_f184 = measure_fwhm(wfi.calc_psf(oversample=6))

    assert 4.0 > fwhm_f184 / fwhm_f062 > 2.0


def test_WFI_pupil_controller():
    wfi = roman.WFI()

    for detector in wfi.detector_list:
        wfi.detector = detector

        assert os.path.isfile(pupil_path(wfi)), f'Pupil file missing: {pupil_path(wfi)}'

        # Test detector change was successful
        assert wfi.detector == detector, 'WFI detector was not set correctly'
        assert wfi.pupil == pupil_path(wfi), 'pupil path was not set correctly'

        # Test pupil mask lock/unlock
        for mask in wfi.pupil_mask_list:
            # test lock
            wfi.lock_pupil_mask(mask)

            assert wfi.pupil == pupil_path(wfi, mask), 'Pupil path was not set correctly'

            # introduce differing filter to modify
            wfi.filter = 'PRISM' if mask != 'PRISM' else 'F062'

            assert wfi._pupil_controller._pupil_mask == wfi.pupil_mask, 'Pupil mask was not set correctly'

            # test unlock
            wfi.unlock_pupil_mask()

            assert wfi.pupil == pupil_path(wfi), 'Pupil mask unlock failed'

        assert wfi._pupil_controller._auto_pupil, 'Pupil is locked and should not be'
        assert wfi._pupil_controller._auto_pupil_mask, 'Pupil mask is locked and should not be'

        # Test pupil lock/unlock
        with pytest.raises(FileNotFoundError):
            assert wfi.lock_pupil('file_that_does_not_exist.fits'), 'FileNotFoundError was not raised'

        this_file = __file__
        wfi.lock_pupil(this_file)
        assert wfi.pupil == this_file, 'Pupil did not lock to proper file.'

        wfi.unlock_pupil()
        assert wfi.pupil == pupil_path(wfi), 'Pupil unlock failed.'

        assert wfi._pupil_controller._auto_pupil, 'Pupil is locked and should  not be'
        assert wfi._pupil_controller._auto_pupil_mask, 'Pupil mask is locked and should not be'

        # Test effect of changing the filter on pupil path
        for fltr in wfi.filter_list:
            wfi.filter = fltr

            assert wfi.pupil == pupil_path(wfi), f'Pupil was not set to correct value for filter {fltr}'

    # Test persistence of pupil and pupil mask locks through a PSF calculation
    wfi2 = roman.WFI()
    wfi2.detector = detector
    valid_pos = (4000, 1000)
    wfi2.detector_position = valid_pos

    wfi2.filter = 'F129'
    wfi2.lock_pupil_mask('GRISM')
    wfi2.filter = 'F129'
    assert wfi2.pupil == pupil_path(wfi2, 'GRISM'), 'Pupil path was not set correctly'
    wfi2.calc_psf(monochromatic=1.3e-6, fov_pixels=4)

    assert wfi.pupil_mask == 'GRISM', 'Pupil mask changed during PSF calculation'
    assert wfi2.pupil == pupil_path(wfi2, 'GRISM'), 'Pupil path changed during PSF calculation'


def test_WFI_detector_position_setter():
    wfi = roman.WFI()
    wfi.detector = 'WFI01'
    valid_pos = (4000, 1000)
    wfi.detector_position = valid_pos
    assert wfi._detectors[wfi._detector].field_position == valid_pos, (
        'Setting field position through Instrument.detector_position did not update field_position '
        "for the detector's aberration optic"
    )
    assert wfi.detector_position == valid_pos, "`detector_position` getter doesn't reflect " 'assignment to setter'


def test_WFI_includes_aberrations():
    wfi = roman.WFI()
    wfi.detector = 'WFI01'
    osys = wfi.get_optical_system()
    assert roman.FieldDependentAberration in [type(o) for o in osys], (
        'Roman WFI optical system should include plane for field dependent aberration virtual optic'
    )


def test_swapping_modes(wfi=None):
    if wfi is None:
        wfi = roman.WFI()

    tests = [
        # [filter, mode, pupil_file]
        ['F146', 'F146', pupil_path],
        ['F213', 'F213', pupil_path],
        [PRISM_FILTERS[0], PRISM_FILTERS[0], pupil_path],
        [GRISM_FILTERS[0], GRISM_FILTERS[0], pupil_path],
    ]

    for test_filter, test_mode, test_pupil in tests:
        wfi.filter = test_filter

        fail_str = f"failed on {test_filter}, {test_mode}, " f"{test_pupil(wfi).split('/')[-1]}"

        assert wfi.filter == test_filter, fail_str
        assert wfi.mode == test_mode, fail_str
        assert wfi._current_aberration_file == wfi._aberration_files[test_mode], fail_str
        assert wfi.pupil == test_pupil(wfi), fail_str


def test_custom_aberrations():
    wfi = roman.WFI()

    # Use GRISM0 aberration_file for testing
    test_aberration_file = wfi._aberration_files['GRISM0']

    # Test override
    # -------------
    wfi.lock_aberrations(test_aberration_file)

    for fltr in wfi.filter_list:
        wfi.filter = fltr
        assert wfi._current_aberration_file == test_aberration_file, 'Filter change caused override to fail'

    # Test Release Override
    # ---------------------
    wfi.unlock_aberrations()
    assert wfi._aberration_files['custom'] is None, 'Custom aberration file not deleted on override release.'
    test_swapping_modes(wfi)


def test_WFI_limits_interpolation_range():
    wfi = roman.WFI()
    det = wfi._detectors['WFI01']
    det.get_aberration_terms(1.29e-6)
    det.field_position = (0, 0)
    det.get_aberration_terms(1.29e-6)

    with pytest.raises(ValueError) as excinfo:
        det.field_position = (500000, 0)
    assert 'Requested pixel_x position' in str(
        excinfo.value
    ), 'FieldDependentAberration did not error on out-of-bounds field point'

    with pytest.raises(ValueError) as excinfo:
        det.field_position = (-1, 0)
    assert 'Requested pixel_x position' in str(
        excinfo.value
    ), 'FieldDependentAberration did not error on out-of-bounds field point'

    with pytest.raises(ValueError) as excinfo:
        det.field_position = (0, 500000)
    assert 'Requested pixel_y position' in str(
        excinfo.value
    ), 'FieldDependentAberration did not error on out-of-bounds field point'

    with pytest.raises(ValueError) as excinfo:
        det.field_position = (0, -1)
    assert 'Requested pixel_y position' in str(
        excinfo.value
    ), 'FieldDependentAberration did not error on out-of-bounds field point'

    det.field_position = (2048, 2048)

    # Get min and max valid wavelengths from aberration file
    zern = Table.read(wfi._aberration_files[wfi.mode], format='ascii.csv')
    min_wv = zern['wave'][0] * 1e-9  # convert from nm to meter
    max_wv = zern['wave'][-1] * 1e-9

    # Test that get_aberration_terms() uses an approximated wavelength when
    # called with an out-of-bounds wavelength.
    too_lo_wv = min_wv * 0.9
    too_hi_wv = max_wv / 0.9
    valid_wv = np.mean([min_wv, max_wv])

    assert allclose(
        det.get_aberration_terms(min_wv), det.get_aberration_terms(too_lo_wv)
    ), 'Aberration below wavelength range did not return closest value.'

    assert allclose(
        det.get_aberration_terms(max_wv), det.get_aberration_terms(too_hi_wv)
    ), 'Aberration above wavelength range did not return closest value.'

    # Test border pixels outside the ref data. In Cycle 10, (32, 0) is the first
    # pixel, so we check if (0, 0) is approximated to it as the nearest point.
    det.field_position = (0, 0)
    coefficients_outlier = det.get_aberration_terms(valid_wv)

    det.field_position = (32, 0)
    coefficients_data = det.get_aberration_terms(valid_wv)

    assert np.allclose(coefficients_outlier, coefficients_data), (
        'nearest point extrapolation ' 'failed for outlier field point'
    )


def test_WFI_auto_aperturename_and_pixelscale():
    wfi = roman.WFI()
    assert wfi.aperturename == "WFI01_FULL", "Aperture name should match detector"
    wfi.detector = 'WFI12'
    assert wfi.aperturename == "WFI12_FULL", "Aperture name should match detector"


    aperture = wfi.siaf[wfi.aperturename]
    assert wfi.pixelscale == (aperture.XSciScale + aperture.YSciScale)/2, "Pixel scale should match the SIAF for that aperturename"


def test_sci_xy_to_fp():
    """Test corners match expectations"""
    corner_data = ( ((0,0), 21),  # lower left
                    ((0, 4096), 25),  # upper left
                    ((4096, 0), 1),  # lower right
                    ((4096, 4096), 5)  # upper right
                  )
    for (x, y), fp in corner_data:
        assert roman._wfi_sci_xy_to_fp(x, y) == fp  # test xy->fp
        assert roman._wfi_sci_xy_to_fp(*roman._wfi_fp_to_sci_xy(fp)) == fp , f'round trip error for fp {fp}'  # test round trip


# ---- svh/newrd fork ----------------------------------------------------------

def test_wfi_ipc_off_by_default():
    """IPC belongs to the detector simulation, so the WFI PSF has none by default."""
    wfi = roman.WFI()
    assert wfi.options['add_ipc'] is False


@pytest.mark.parametrize('det', ['WFI01', 'WFI05', 'WFI18'])
def test_wfi_high_frequency_map_follows_detector(det):
    """Each detector uses its own high-frequency OPD map, also after a filter change."""
    wfi = roman.WFI()
    wfi.detector = det
    n = int(det[3:])
    assert os.path.basename(wfi.pupilopd) == f'RST_WFI{n}_Z210_high_frequency_map.fits'
    wfi.filter = 'GRISM1'
    assert os.path.basename(wfi.pupilopd) == f'RST_WFI{n}_Z210_high_frequency_map.fits'


def test_wfi_ipc_matches_explicit_option():
    """add_ipc=True still reproduces the 2.3.0 default; False leaves OVERDIST unconvolved."""
    wfi = roman.WFI()
    wfi.filter = 'F158'
    off = wfi.calc_psf(fov_pixels=11, oversample=2, nlambda=1)
    wfi.options['add_ipc'] = True
    on = wfi.calc_psf(fov_pixels=11, oversample=2, nlambda=1)
    d = 'OVERDIST'
    assert not np.allclose(off[d].data, on[d].data)
    # IPC moves light from the core into the neighbours
    assert on[d].data.max() < off[d].data.max()


@pytest.mark.parametrize('first, second', [('F158', 'F087'), ('F158', 'GRISM1'), ('GRISM0', 'PRISM')])
def test_wfi_filter_change_keeps_field_position(first, second):
    """Changing the filter after setting a position keeps the Zernikes at that position."""
    pos = (300, 3800)
    wfi = roman.WFI()
    wfi.detector = 'WFI05'
    wfi.filter = first
    wfi.detector_position = pos
    wfi.filter = second
    assert wfi._detectors['WFI05'].field_position == pos

    ref = roman.WFI()
    ref.detector = 'WFI05'
    ref.filter = second
    ref.detector_position = pos
    np.testing.assert_array_equal(wfi._detectors['WFI05'].get_aberration_terms(1.2e-6),
                                  ref._detectors['WFI05'].get_aberration_terms(1.2e-6))
    kw = dict(fov_pixels=11, oversample=2, monochromatic=1.2e-6)
    np.testing.assert_allclose(wfi.calc_psf(**kw)['OVERSAMP'].data, ref.calc_psf(**kw)['OVERSAMP'].data,
                               rtol=0, atol=1e-12)


def test_wfi_unlock_aberrations_keeps_field_position():
    """Reloading the default aberrations also keeps the current position."""
    pos = (300, 3800)
    wfi = roman.WFI()
    wfi.detector = 'WFI05'
    wfi.detector_position = pos
    wfi.lock_aberrations(wfi._aberration_files[wfi.mode])
    assert wfi._detectors['WFI05'].field_position == pos
    wfi.unlock_aberrations()
    assert wfi._detectors['WFI05'].field_position == pos


def test_wfi_aberration_table_not_modified():
    """svh/newrd: omitting Z1-3 must not zero them in the stored Zernike table."""
    wfi = roman.WFI()
    wfi.filter = 'F158'
    wfi.detector = 'WFI05'
    aber = wfi._detectors['WFI05']
    interp = next(iter(aber._wavelength_interpolators.values()))
    wl = interp._wavelengths[0]
    stored = interp._aberration_terms.copy()
    assert np.any(stored[0, :3] != 0)

    # exact field point (short path) and an off-grid one, both at a tabulated wavelength
    for fp in list(aber._wavelength_interpolators)[:1] + [(2000, 2000)]:
        aber.field_position = fp
        terms = aber.get_aberration_terms(wl)
        assert np.all(terms[:3] == 0)
    np.testing.assert_array_equal(interp._aberration_terms, stored)

    aber._omit_piston_tip_tilt = False
    aber.field_position = list(aber._wavelength_interpolators)[0]
    np.testing.assert_array_equal(aber.get_aberration_terms(wl), stored[0])
