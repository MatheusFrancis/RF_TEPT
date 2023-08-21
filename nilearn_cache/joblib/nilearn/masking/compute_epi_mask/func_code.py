# first line: 212
@_utils.fill_doc
def compute_epi_mask(
    epi_img,
    lower_cutoff=0.2,
    upper_cutoff=0.85,
    connected=True,
    opening=2,
    exclude_zeros=False,
    ensure_finite=True,
    target_affine=None,
    target_shape=None,
    memory=None,
    verbose=0,
):
    """Compute a brain mask from :term:`fMRI` data in 3D or \
    4D :class:`numpy.ndarray`.

    This is based on an heuristic proposed by T.Nichols:
    find the least dense point of the histogram, between fractions
    ``lower_cutoff`` and ``upper_cutoff`` of the total image histogram.

    .. note::

        In case of failure, it is usually advisable to
        increase ``lower_cutoff``.

    Parameters
    ----------
    epi_img : Niimg-like object
        See :ref:`extracting_data`.
        :term:`EPI` image, used to compute the mask.
        3D and 4D images are accepted.

        .. note::
            If a 3D image is given, we suggest to use the mean image.

    %(lower_cutoff)s
        Default=0.2.
    %(upper_cutoff)s
        Default=0.85.
    %(connected)s
        Default=True.
    %(opening)s
        Default=2.
    ensure_finite : :obj:`bool`
        If ensure_finite is True, the non-finite values (NaNs and infs)
        found in the images will be replaced by zeros
        Default=True.

    exclude_zeros : :obj:`bool`, optional
        Consider zeros as missing values for the computation of the
        threshold. This option is useful if the images have been
        resliced with a large padding of zeros.
        Default=False.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(memory)s
    %(verbose0)s

    Returns
    -------
    mask : :class:`nibabel.nifti1.Nifti1Image`
        The brain mask (3D image).
    """
    if verbose > 0:
        print("EPI mask computation")

    # Delayed import to avoid circular imports
    from .image.image import _compute_mean

    mean_epi, affine = cache(_compute_mean, memory)(
        epi_img,
        target_affine=target_affine,
        target_shape=target_shape,
        smooth=(1 if opening else False),
    )

    if ensure_finite:
        # Get rid of memmapping
        mean_epi = _utils.as_ndarray(mean_epi)
        # SPM tends to put NaNs in the data outside the brain
        mean_epi[np.logical_not(np.isfinite(mean_epi))] = 0
    sorted_input = np.sort(np.ravel(mean_epi))
    if exclude_zeros:
        sorted_input = sorted_input[sorted_input != 0]
    lower_cutoff = int(np.floor(lower_cutoff * len(sorted_input)))
    upper_cutoff = min(
        int(np.floor(upper_cutoff * len(sorted_input))), len(sorted_input) - 1
    )

    delta = (
        sorted_input[lower_cutoff + 1 : upper_cutoff + 1]
        - sorted_input[lower_cutoff:upper_cutoff]
    )
    ia = delta.argmax()
    threshold = 0.5 * (
        sorted_input[ia + lower_cutoff] + sorted_input[ia + lower_cutoff + 1]
    )

    mask = mean_epi >= threshold

    mask, affine = _post_process_mask(
        mask,
        affine,
        opening=opening,
        connected=connected,
        warning_msg="Are you sure that input "
        "data are EPI images not detrended. ",
    )
    return new_img_like(epi_img, mask, affine)
